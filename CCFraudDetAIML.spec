# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['CCFraudDetAIML.py'],
    pathex=[],
    binaries=[],
    datas=[('custominfo.plist', '.'), ('aiml.icns', '.')],  # Include your icon file if needed
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CCFraudDetAIML',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='1.0',  # Set your version number here
    description='Credit Card Fraud Detection using AIML',  # Set your app description
    icon='aiml.icns',  # Include your icon file if needed
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CCFraudDetAIML',
)
