#!/bin/bash

WALLPAPER_DIR="/home/4n4tm4n/wallpapers"

selected=$(find "$WALLPAPER_DIR" -type f | fzf)

if [ -n "$selected" ]; then
    swww img "$selected"
    sed -i "s|swww img.*$|swww img $selected|" /home/4n4tm4n/.config/hypr/config/autostart.conf
fi

