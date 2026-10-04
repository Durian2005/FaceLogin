' Face Login silent launcher
' Paths are built with FileSystemObject.BuildPath, so a missing backslash
' (the previous bug) cannot happen again.
Option Explicit

Dim fso, sh, root, py, runner
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

root   = fso.GetParentFolderName(WScript.ScriptFullName)
py     = fso.BuildPath(fso.BuildPath(fso.BuildPath(root, ".venv"), "Scripts"), "pythonw.exe")
runner = fso.BuildPath(fso.BuildPath(root, "app"), "run.py")

If Not fso.FileExists(py) Then
  MsgBox "Python venv not found:" & vbCrLf & py & vbCrLf & vbCrLf & _
         "Please double-click the .bat launcher once to create it.", 16, "Face Login"
  WScript.Quit 1
End If

If Not fso.FileExists(runner) Then
  MsgBox "app\run.py not found:" & vbCrLf & runner & vbCrLf & vbCrLf & _
         "Please check the project folder.", 16, "Face Login"
  WScript.Quit 1
End If

sh.CurrentDirectory = root
sh.Run """" & py & """ """ & runner & """", 0, False
