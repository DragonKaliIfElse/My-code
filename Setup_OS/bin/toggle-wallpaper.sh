#!/bin/bash

WALLPAPER_DIR="/home/4n4tm4n/wallpapers"
FILES=$(find "$WALLPAPER_DIR" -type f)
NUMBER_OF_WALLPAPERS=$(echo "$FILES" | wc -l)
N_SELECTED=$(((RANDOM % NUMBER_OF_WALLPAPERS) + 1))
F_SELECTED=$(echo "$FILES" | head -n "$N_SELECTED" | tail -n 1)

swww img "$F_SELECTED"
