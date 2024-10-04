#!/bin/sh
# Create a folder (named dmg) to prepare our DMG in (if it doesn't already exist).
mkdir -p dist/dmg
# Empty the dmg folder.
rm -r dist/dmg/*
# Copy the app bundle to the dmg folder.
cp -r "dist/Hello World.app" dist/dmg
# If the DMG already exists, delete it.
test -f "dist/Hello World.dmg" && rm "dist/Hello World.dmg"
create-dmg \
  --volname "CCFraudDetAIML" \
  --volicon "aiml.icns" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --icon "CCFraudDetAIML.app" 175 120 \
  --hide-extension "CCFraudDetAIML.app" \
  --app-drop-link 425 120 \
  "dist/CCFraudDetAIML.dmg" \
  "dist/dmg/"
