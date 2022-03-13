# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

resources = (("bitbug_favicon.ico", "."), ("mmod_human_face_detector.dat", "."),
 ("Basic_Epoch_3_Accuracy_0.93.pth", "."), ("shape_predictor_68_face_landmarks.dat", ".")
 , ("forword11.png", "."), ("forword21.png", "."), ("left1.png", ".")
 , ("left2.png", "."), ("up1.png", "."), ("up2.png", "."),("right1.png", "."), ("right2.png", "."))


a = Analysis(['blink.py'],
             pathex=['/home/e1009/intelligent_wheelchair_gui'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='blink',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='blink')
