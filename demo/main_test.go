package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRestorePostprocessHistory(t *testing.T) {
	directory := t.TempDir()
	write := func(id, manifest string) {
		t.Helper()
		path := filepath.Join(directory, id)
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "manifest.json"), []byte(manifest), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("legacy", `{"id":"legacy","mode":"rig","state":"done"}`)
	write("raw", `{"id":"raw","mode":"rig","state":"done","postprocessed":false}`)
	write("postprocessed", `{"id":"postprocessed","mode":"rig","state":"done","postprocessed":true}`)
	write("old-fit", `{"id":"old-fit","mode":"skin","inputLayout":"separate","state":"done"}`)
	write("global-fit", `{"id":"global-fit","mode":"skin","inputLayout":"separate","state":"done","fitMode":"global"}`)
	write("articulated-fit", `{"id":"articulated-fit","mode":"skin","inputLayout":"separate","state":"done","fitMode":"articulated"}`)

	records := restore(directory)
	if !records["legacy"].Postprocessed {
		t.Fatal("legacy generation should retain the old always-postprocessed meaning")
	}
	if records["raw"].Postprocessed {
		t.Fatal("explicit raw generation was restored as postprocessed")
	}
	if !records["postprocessed"].Postprocessed {
		t.Fatal("explicit postprocessed generation was restored as raw")
	}
	if records["old-fit"].FitMode != "legacy-articulated" {
		t.Fatal("generation from the old arm-reposing implementation lost its history label")
	}
	if records["global-fit"].FitMode != "global" || records["articulated-fit"].FitMode != "articulated" {
		t.Fatal("explicit skeleton fit mode was not preserved")
	}
}
