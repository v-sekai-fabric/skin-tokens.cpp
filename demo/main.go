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

type record struct {
	ID         string `json:"id"`
	MeshName   string `json:"meshName"`
	MotionName string `json:"motionName"`
	State      string `json:"state"`
	Error      string `json:"error,omitempty"`
	CreatedAt  int64  `json:"createdAt"`
	FinishedAt int64  `json:"finishedAt,omitempty"`
	Learned    bool   `json:"learned"`
	TargetRig  string `json:"targetRig"`
	MeshPath   string `json:"-"`
	MotionPath string `json:"-"`
}

type source struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Kind  string `json:"kind"`
	Path  string `json:"-"`
	Label string `json:"label,omitempty"`
}

type server struct {
	mu          sync.RWMutex
	records     map[string]*record
	sources     map[string]source
	queue       chan string
	data        string
	cli         string
	model       string
	device      string
	remesher    string
	targetQuads int
	timeout     time.Duration
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

func (s *server) persist(value *record) {
	data, _ := json.MarshalIndent(value, "", "  ")
	_ = os.WriteFile(filepath.Join(s.data, value.ID, "manifest.json"), append(data, '\n'), 0o600)
}

func (s *server) worker() {
	for id := range s.queue {
		s.mu.Lock()
		value := s.records[id]
		s.mu.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
		meshPath := value.MeshPath
		var combined []byte
		var err error
		if strings.EqualFold(filepath.Ext(meshPath), ".t2mesh") {
			s.mu.Lock()
			value.State = "remeshing"
			s.persist(value)
			s.mu.Unlock()
			if s.remesher == "" {
				err = errors.New("raw T2MESH input requires trellis2cpp mesh2glb; start the demo with --trellis-mesh2glb PATH")
			} else {
				meshPath = filepath.Join(s.data, id, "remeshed.glb")
				command := exec.CommandContext(ctx, s.remesher, value.MeshPath, meshPath,
					"2048", "--print", "0.5", "0.0166667", "--quad", fmt.Sprint(s.targetQuads))
				combined, err = command.CombinedOutput()
			}
		}
		if err == nil {
			s.mu.Lock()
			value.State = "running"
			s.persist(value)
			s.mu.Unlock()
		}
		output := filepath.Join(s.data, id, "animation.glb")
		if err == nil {
			command := exec.CommandContext(ctx, s.cli, "bind", s.model, meshPath, value.MotionPath,
				output, "--device", s.device, "--supplied-skeleton", "--target-rig", value.TargetRig,
				"--postprocess")
			combined, err = command.CombinedOutput()
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
	s.mu.RLock()
	value, ok := s.sources[id]
	s.mu.RUnlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	if strings.ToLower(filepath.Ext(value.Path)) != ".glb" {
		http.Error(w, "browser preview is available for GLB sources only", http.StatusUnsupportedMediaType)
		return
	}
	file, err := os.Open(value.Path)
	if err != nil {
		http.Error(w, "source is no longer available", http.StatusNotFound)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 512<<20 {
		http.Error(w, "source is not a bounded regular file", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "model/gltf-binary")
	w.Header().Set("Cache-Control", "no-store")
	http.ServeContent(w, r, filepath.Base(value.Path), info.ModTime(), file)
}

func (s *server) bind(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 520<<20)
	if err := r.ParseMultipartForm(2 << 20); err != nil {
		http.Error(w, "invalid upload: "+err.Error(), 400)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	id := identifier()
	directory := filepath.Join(s.data, id)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	targetRig := r.FormValue("targetRig")
	if targetRig == "" {
		targetRig = "soma30"
	}
	if targetRig != "soma30" && targetRig != "mixamo52" {
		http.Error(w, "target rig must be soma30 or mixamo52", 400)
		return
	}
	value := &record{ID: id, State: "queued", CreatedAt: time.Now().UnixMilli(), TargetRig: targetRig}
	resolve := func(kind string) (string, string, error) {
		if sourceID := r.FormValue(kind + "Source"); sourceID != "" {
			item, ok := s.sources[sourceID]
			if !ok || item.Kind != kind {
				return "", "", errors.New("unknown local source")
			}
			return item.Path, item.Name, nil
		}
		headerList := r.MultipartForm.File[kind]
		if len(headerList) != 1 {
			return "", "", errors.New("choose one " + kind + " GLB")
		}
		name := safeName(headerList[0].Filename)
		extension := strings.ToLower(filepath.Ext(name))
		if extension != ".glb" && !(kind == "mesh" && extension == ".t2mesh") {
			return "", "", errors.New("mesh uploads must be .glb or .t2mesh; motions must be .glb")
		}
		path := filepath.Join(directory, kind+extension)
		if err := saveUpload(headerList[0], path); err != nil {
			return "", "", err
		}
		return path, name, nil
	}
	var err error
	value.MeshPath, value.MeshName, err = resolve("mesh")
	if err != nil {
		_ = os.RemoveAll(directory)
		http.Error(w, err.Error(), 400)
		return
	}
	value.MotionPath, value.MotionName, err = resolve("motion")
	if err != nil {
		_ = os.RemoveAll(directory)
		http.Error(w, err.Error(), 400)
		return
	}
	s.mu.Lock()
	s.records[id] = value
	s.persist(value)
	s.mu.Unlock()
	select {
	case s.queue <- id:
		writeJSON(w, value)
	default:
		http.Error(w, "inference queue is full", 503)
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

func scanSources(root, kind string, output map[string]source) {
	if root == "" {
		return
	}
	_ = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			if entry != nil && entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		extension := strings.ToLower(filepath.Ext(path))
		if entry.IsDir() || (extension != ".glb" && !(kind == "mesh" && extension == ".t2mesh")) {
			return nil
		}
		if kind == "motion" && filepath.Base(path) != "animation.glb" {
			return nil
		}
		absolute, err := filepath.Abs(path)
		if err != nil {
			return nil
		}
		info, err := entry.Info()
		if err != nil || info.Size() > 512<<20 {
			return nil
		}
		id := kind + "-" + identifier()
		label := filepath.Base(filepath.Dir(path))
		if kind == "motion" {
			if text, e := os.ReadFile(filepath.Join(filepath.Dir(path), "prompt.txt")); e == nil {
				label = strings.TrimSpace(string(text))
			}
		}
		output[id] = source{ID: id, Name: filepath.Base(path), Kind: kind, Path: absolute, Label: label}
		return nil
	})
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
		if json.Unmarshal(data, &value) == nil {
			if value.TargetRig == "" {
				value.TargetRig = "soma30"
			}
			if value.State == "running" || value.State == "remeshing" || value.State == "queued" {
				value.State = "error"
				value.Error = "server stopped before completion"
			}
			values[value.ID] = &value
		}
	}
	return values
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8095", "listen address")
	data := flag.String("data", "demo-data", "persistent output directory")
	cli := flag.String("cli", "build/release/bin/skintokens-cli", "skintokens-cli path")
	model := flag.String("model", "models/skintokens-f32", "GGUF bundle")
	device := flag.String("device", "vulkan", "auto, cpu, or vulkan")
	remesher := flag.String("trellis-mesh2glb", "", "trellis2cpp mesh2glb path (required for T2MESH inputs)")
	targetQuads := flag.Int("target-quads", 20000, "quad-remesh density for T2MESH inputs")
	kimodo := flag.String("kimodo", "", "Kimodo gallery directory (empty disables it)")
	trellis := flag.String("trellis", "", "trellis2cpp gallery directory (empty disables it)")
	giraffeInput := flag.String("giraffe-input", "", "official SkinTokens giraffe input GLB (empty disables it)")
	giraffeResult := flag.String("giraffe-result", "", "captured official SkinTokens giraffe result GLB (empty disables it)")
	timeout := flag.Duration("timeout", 30*time.Minute, "binding timeout")
	flag.Parse()
	if *targetQuads < 100 || *targetQuads > 500000 {
		log.Fatal("target-quads must be between 100 and 500000")
	}
	if err := os.MkdirAll(*data, 0o700); err != nil {
		log.Fatal(err)
	}
	s := &server{records: restore(*data), sources: map[string]source{}, queue: make(chan string, 16),
		data: *data, cli: *cli, model: *model, device: *device, remesher: *remesher,
		targetQuads: *targetQuads, timeout: *timeout}
	scanSources(*kimodo, "motion", s.sources)
	scanSources(*trellis, "mesh", s.sources)
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
	mux.HandleFunc("GET /api/sources", s.listSources)
	mux.HandleFunc("GET /api/sources/", s.sourceFile)
	mux.HandleFunc("POST /api/bind", s.bind)
	mux.HandleFunc("GET /api/jobs/", s.item)
	mux.Handle("GET /files/", http.StripPrefix("/files/", http.FileServer(http.Dir(*data))))
	log.Printf("SkinTokens demo listening on http://%s (%d local sources)", *listen, len(s.sources))
	if err := http.ListenAndServe(*listen, mux); err != nil {
		log.Fatal(fmt.Errorf("server: %w", err))
	}
}
