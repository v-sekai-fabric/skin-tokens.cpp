package main

import (
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed web
var web embed.FS

type glbInfo struct {
	HasMesh         bool    `json:"hasMesh"`
	HasSkin         bool    `json:"hasSkin"`
	HasSkeleton     bool    `json:"hasSkeleton"`
	HasAnimation    bool    `json:"hasAnimation"`
	JointCount      int     `json:"jointCount"`
	FrameCount      int     `json:"frameCount"`
	FramesPerSecond float64 `json:"framesPerSecond"`
	RigKind         string  `json:"rigKind"`
}

type record struct {
	ID            string `json:"id"`
	Mode          string `json:"mode"`
	InputLayout   string `json:"inputLayout,omitempty"`
	MeshName      string `json:"meshName"`
	SkeletonName  string `json:"skeletonName,omitempty"`
	State         string `json:"state"`
	Error         string `json:"error,omitempty"`
	CreatedAt     int64  `json:"createdAt"`
	FinishedAt    int64  `json:"finishedAt,omitempty"`
	Learned       bool   `json:"learned"`
	HasAnimation  bool   `json:"hasAnimation"`
	RigKind       string `json:"rigKind,omitempty"`
	Retargeted    bool   `json:"retargeted,omitempty"`
	Postprocessed bool   `json:"postprocessed"`
	FitMode       string `json:"fitMode"`
	OutputFile    string `json:"outputFile"`

	// Legacy fields keep existing history readable after upgrading the demo.
	MotionName   string `json:"motionName,omitempty"`
	TargetRig    string `json:"targetRig,omitempty"`
	MeshPath     string `json:"-"`
	SkeletonPath string `json:"-"`
}

type source struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Kind  string `json:"kind"`
	Path  string `json:"-"`
	Label string `json:"label,omitempty"`
}

type server struct {
	mu      sync.RWMutex
	records map[string]*record
	sources map[string]source
	queue   chan string
	data    string
	cli     string
	model   string
	device  string
	timeout time.Duration
}

func identifier() string {
	var value [8]byte
	if _, err := rand.Read(value[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(value[:])
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(value)
}

func safeName(value string) string {
	value = filepath.Base(strings.TrimSpace(value))
	if value == "." || value == "" {
		return "asset.glb"
	}
	return value
}

func saveUpload(header *multipart.FileHeader, path string) error {
	input, err := header.Open()
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer output.Close()
	_, err = io.Copy(output, input)
	return err
}

func oneUpload(r *http.Request, field, directory string) (string, string, error) {
	values := r.MultipartForm.File[field]
	if len(values) != 1 {
		return "", "", errors.New("choose one " + field + " GLB")
	}
	name := safeName(values[0].Filename)
	if !strings.EqualFold(filepath.Ext(name), ".glb") {
		return "", "", errors.New(field + " must be a GLB file")
	}
	path := filepath.Join(directory, field+".glb")
	if err := saveUpload(values[0], path); err != nil {
		return "", "", err
	}
	return path, name, nil
}

func (s *server) persist(value *record) {
	data, _ := json.MarshalIndent(value, "", "  ")
	_ = os.WriteFile(filepath.Join(s.data, value.ID, "manifest.json"), append(data, '\n'), 0o600)
}

func (s *server) inspect(ctx context.Context, path string) (glbInfo, error) {
	command := exec.CommandContext(ctx, s.cli, "glb-info", path)
	combined, err := command.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(combined))
		if message == "" {
			message = err.Error()
		}
		return glbInfo{}, errors.New(message)
	}
	var value glbInfo
	if err := json.Unmarshal(combined, &value); err != nil {
		return glbInfo{}, fmt.Errorf("invalid inspector response: %w", err)
	}
	return value, nil
}

func (s *server) worker() {
	for id := range s.queue {
		s.mu.RLock()
		value := s.records[id]
		s.mu.RUnlock()
		if value == nil {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
		s.mu.Lock()
		value.State = "running"
		s.persist(value)
		s.mu.Unlock()

		output := filepath.Join(s.data, id, value.OutputFile)
		hasAnimation, rigKind := value.HasAnimation, value.RigKind
		var command *exec.Cmd
		if value.Mode == "rig" {
			arguments := []string{"rig", s.model, value.MeshPath, output, "--device", s.device}
			if value.Postprocessed {
				arguments = append(arguments, "--postprocess")
			}
			command = exec.CommandContext(ctx, s.cli, arguments...)
		} else {
			arguments := []string{"skin", s.model, value.MeshPath, value.SkeletonPath, output,
				"--device", s.device}
			if value.Postprocessed {
				arguments = append(arguments, "--postprocess")
			}
			if value.InputLayout == "embedded" {
				arguments = append(arguments, "--fit", "none")
			} else {
				fit := value.FitMode
				if fit != "articulated" {
					fit = "global"
				}
				arguments = append(arguments, "--fit", fit)
			}
			if value.Retargeted {
				arguments = append(arguments, "--retarget-soma-to-mixamo52")
			}
			command = exec.CommandContext(ctx, s.cli, arguments...)
		}
		combined, err := command.CombinedOutput()
		if err == nil {
			if info, inspectErr := s.inspect(ctx, output); inspectErr == nil {
				hasAnimation = info.HasAnimation
				rigKind = info.RigKind
			} else {
				err = inspectErr
				combined = []byte(inspectErr.Error())
			}
		}
		cancel()

		s.mu.Lock()
		value.FinishedAt = time.Now().UnixMilli()
		if err != nil {
			value.State = "error"
			value.Error = strings.TrimSpace(string(combined))
			if value.Error == "" {
				value.Error = err.Error()
			}
			if len(value.Error) > 6000 {
				value.Error = value.Error[len(value.Error)-6000:]
			}
		} else {
			value.State = "done"
			value.Learned = true
			value.HasAnimation = hasAnimation
			value.RigKind = rigKind
		}
		s.persist(value)
		s.mu.Unlock()
	}
}

func (s *server) history(w http.ResponseWriter, _ *http.Request) {
	s.mu.RLock()
	values := make([]record, 0, len(s.records))
	for _, value := range s.records {
		values = append(values, *value)
	}
	s.mu.RUnlock()
	sort.Slice(values, func(i, j int) bool { return values[i].CreatedAt > values[j].CreatedAt })
	writeJSON(w, values)
}

func (s *server) inspectUpload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 514<<20)
	if err := r.ParseMultipartForm(2 << 20); err != nil {
		http.Error(w, "invalid upload: "+err.Error(), http.StatusBadRequest)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	values := r.MultipartForm.File["file"]
	if len(values) != 1 || !strings.EqualFold(filepath.Ext(values[0].Filename), ".glb") {
		http.Error(w, "choose one GLB file", http.StatusBadRequest)
		return
	}
	temporary, err := os.CreateTemp(s.data, "inspect-*.glb")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	path := temporary.Name()
	_ = temporary.Close()
	_ = os.Remove(path)
	defer os.Remove(path)
	if err := saveUpload(values[0], path); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()
	info, err := s.inspect(ctx, path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, info)
}

func (s *server) generate(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1028<<20)
	if err := r.ParseMultipartForm(2 << 20); err != nil {
		http.Error(w, "invalid upload: "+err.Error(), http.StatusBadRequest)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	mode, layout := r.FormValue("mode"), r.FormValue("layout")
	if mode != "rig" && mode != "skin" {
		http.Error(w, "mode must be rig or skin", 400)
		return
	}
	if mode == "skin" && layout != "embedded" && layout != "separate" {
		http.Error(w, "skin layout must be embedded or separate", 400)
		return
	}
	retarget := r.FormValue("retargetSomaToMixamo") == "1"
	postprocess := r.FormValue("postprocess") == "1"
	articulatedFit := r.FormValue("articulatedFit") == "1"
	if mode != "skin" && retarget {
		http.Error(w, "retargeting is available in skin-only mode", 400)
		return
	}
	if articulatedFit && (mode != "skin" || layout != "separate") {
		http.Error(w, "articulated fitting requires separate mesh and skeleton files", 400)
		return
	}

	id := identifier()
	directory := filepath.Join(s.data, id)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.RemoveAll(directory)
		}
	}()
	fitMode := "none"
	if mode == "skin" && layout == "separate" {
		fitMode = "global"
		if articulatedFit {
			fitMode = "articulated"
		}
	}
	value := &record{ID: id, Mode: mode, InputLayout: layout, State: "queued",
		CreatedAt: time.Now().UnixMilli(), Retargeted: retarget, Postprocessed: postprocess,
		FitMode: fitMode, OutputFile: "result.glb"}
	var err error
	if mode == "rig" {
		value.MeshPath, value.MeshName, err = oneUpload(r, "mesh", directory)
	} else if layout == "embedded" {
		value.MeshPath, value.MeshName, err = oneUpload(r, "asset", directory)
		value.SkeletonPath, value.SkeletonName = value.MeshPath, value.MeshName
	} else {
		value.MeshPath, value.MeshName, err = oneUpload(r, "mesh", directory)
		if err == nil {
			value.SkeletonPath, value.SkeletonName, err = oneUpload(r, "skeleton", directory)
		}
	}
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	meshInfo, err := s.inspect(ctx, value.MeshPath)
	if err != nil || !meshInfo.HasMesh {
		if err == nil {
			err = errors.New("mesh upload contains no mesh")
		}
		http.Error(w, err.Error(), 400)
		return
	}
	if mode == "skin" {
		skeletonInfo, inspectErr := s.inspect(ctx, value.SkeletonPath)
		if inspectErr != nil {
			http.Error(w, inspectErr.Error(), 400)
			return
		}
		if !skeletonInfo.HasSkeleton {
			http.Error(w, "skeleton upload contains no armature", 400)
			return
		}
		if layout == "embedded" && !skeletonInfo.HasSkin {
			http.Error(w, "single-file skin input must be a rigged GLB containing a glTF skin", 400)
			return
		}
		if retarget && skeletonInfo.RigKind != "soma30" {
			http.Error(w, "SOMA-to-Mixamo retargeting requires a detected SOMA30 skeleton", 400)
			return
		}
		value.HasAnimation = skeletonInfo.HasAnimation
		value.RigKind = skeletonInfo.RigKind
	}

	s.mu.Lock()
	s.records[id] = value
	s.persist(value)
	s.mu.Unlock()
	select {
	case s.queue <- id:
		cleanup = false
		writeJSON(w, value)
	default:
		s.mu.Lock()
		delete(s.records, id)
		s.mu.Unlock()
		http.Error(w, "inference queue is full", http.StatusServiceUnavailable)
	}
}

func (s *server) item(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/jobs/")
	s.mu.RLock()
	value := s.records[id]
	s.mu.RUnlock()
	if value == nil {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, value)
}

func (s *server) listSources(w http.ResponseWriter, _ *http.Request) {
	values := make([]source, 0, len(s.sources))
	for _, value := range s.sources {
		values = append(values, value)
	}
	sort.Slice(values, func(i, j int) bool { return values[i].Name < values[j].Name })
	writeJSON(w, values)
}

func (s *server) sourceFile(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/sources/")
	value, ok := s.sources[id]
	if !ok {
		http.NotFound(w, r)
		return
	}
	file, err := os.Open(value.Path)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 512<<20 {
		http.Error(w, "source is not a bounded regular file", 400)
		return
	}
	w.Header().Set("Content-Type", "model/gltf-binary")
	w.Header().Set("Cache-Control", "no-store")
	http.ServeContent(w, r, filepath.Base(value.Path), info.ModTime(), file)
}

func addExampleSource(id, name, kind, path, label string, output map[string]source) {
	if path == "" {
		return
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return
	}
	info, err := os.Stat(absolute)
	if err != nil || !info.Mode().IsRegular() || info.Size() > 512<<20 ||
		!strings.EqualFold(filepath.Ext(absolute), ".glb") {
		return
	}
	output[id] = source{ID: id, Name: name, Kind: kind, Path: absolute, Label: label}
}

func restore(directory string) map[string]*record {
	values := map[string]*record{}
	entries, _ := os.ReadDir(directory)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(directory, entry.Name(), "manifest.json"))
		if err != nil {
			continue
		}
		var value record
		if json.Unmarshal(data, &value) != nil {
			continue
		}
		var fields map[string]json.RawMessage
		fitModePresent := false
		if json.Unmarshal(data, &fields) == nil {
			if _, present := fields["postprocessed"]; !present {
				// Older demo versions always enabled postprocessing.
				value.Postprocessed = true
			}
			_, fitModePresent = fields["fitMode"]
		}
		if value.Mode == "" {
			value.Mode, value.InputLayout = "skin", "separate"
			value.SkeletonName = value.MotionName
			value.Retargeted = value.TargetRig == "mixamo52"
			value.HasAnimation = true
		}
		if value.OutputFile == "" {
			value.OutputFile = "animation.glb"
		}
		if !fitModePresent {
			if value.Mode == "skin" && value.InputLayout == "separate" {
				value.FitMode = "legacy-articulated"
			} else {
				value.FitMode = "none"
			}
		}
		if value.State == "running" || value.State == "remeshing" || value.State == "queued" {
			value.State, value.Error = "error", "server stopped before completion"
		}
		values[value.ID] = &value
	}
	return values
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8095", "listen address")
	data := flag.String("data", "demo-data", "persistent output directory")
	cli := flag.String("cli", "build/release/bin/skintokens-cli", "skintokens-cli path")
	model := flag.String("model", "models/SkinTokens-GGUF/F16", "GGUF bundle")
	device := flag.String("device", "vulkan", "auto, cpu, or vulkan")
	timeout := flag.Duration("timeout", 30*time.Minute, "generation timeout")
	giraffeInput := flag.String("giraffe-input", "", "official SkinTokens giraffe input GLB")
	giraffeResult := flag.String("giraffe-result", "", "captured official SkinTokens giraffe result GLB")
	// Accepted for compatibility with older launch scripts; local-gallery
	// dropdowns and automatic T2MESH remeshing are no longer part of the UI.
	_ = flag.String("kimodo", "", "deprecated")
	_ = flag.String("trellis", "", "deprecated")
	_ = flag.String("trellis-mesh2glb", "", "deprecated")
	_ = flag.Int("target-quads", 20000, "deprecated")
	flag.Parse()
	if err := os.MkdirAll(*data, 0o700); err != nil {
		log.Fatal(err)
	}
	s := &server{records: restore(*data), sources: map[string]source{}, queue: make(chan string, 16),
		data: *data, cli: *cli, model: *model, device: *device, timeout: *timeout}
	addExampleSource("giraffe-input", "Giraffe input", "example-input", *giraffeInput, "Official SkinTokens example", s.sources)
	addExampleSource("giraffe-result", "Giraffe learned rig", "example-result", *giraffeResult, "Official SkinTokens output", s.sources)
	go s.worker()
	assets, err := fs.Sub(web, "web")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServer(http.FS(assets)))
	mux.HandleFunc("GET /api/history", s.history)
	mux.HandleFunc("POST /api/inspect-glb", s.inspectUpload)
	mux.HandleFunc("POST /api/generate", s.generate)
	mux.HandleFunc("GET /api/jobs/", s.item)
	mux.HandleFunc("GET /api/sources", s.listSources)
	mux.HandleFunc("GET /api/sources/", s.sourceFile)
	mux.Handle("GET /files/", http.StripPrefix("/files/", http.FileServer(http.Dir(*data))))
	log.Printf("SkinTokens demo listening on http://%s", *listen)
	if err := http.ListenAndServe(*listen, mux); err != nil {
		log.Fatal(fmt.Errorf("server: %w", err))
	}
}
