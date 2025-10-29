Set oWS = WScript.CreateObject("WScript.Shell")

' Ask user which launcher to use
strMessage = "Choose launcher type:" & vbCrLf & vbCrLf & _
             "YES = PowerShell (Recommended - Better conda support)" & vbCrLf & _
             "NO  = Batch File (Traditional cmd.exe)"

intAnswer = MsgBox(strMessage, vbYesNo + vbQuestion, "CheatGPT Launcher Type")

If intAnswer = vbYes Then
    ' PowerShell version
    strBatchFile = "powershell.exe"
    strArguments = "-ExecutionPolicy Bypass -File ""D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\Start_CheatGPT.ps1"""
    strWorkingDir = "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
Else
    ' Batch file version
    strBatchFile = "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\Start_CheatGPT.bat"
    strArguments = ""
    strWorkingDir = "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
End If

' Define paths
strIconFile = "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\web_app\static\favicon.ico"
strDesktop = oWS.SpecialFolders("Desktop")
strShortcut = strDesktop & "\CheatGPT System.lnk"

' Create shortcut on desktop
Set oLink = oWS.CreateShortcut(strShortcut)
oLink.TargetPath = strBatchFile
oLink.Arguments = strArguments
oLink.WorkingDirectory = strWorkingDir
oLink.Description = "Start CheatGPT Detection System"
oLink.IconLocation = strIconFile
oLink.WindowStyle = 1
oLink.Save

' Show success message
If intAnswer = vbYes Then
    strType = "PowerShell"
Else
    strType = "Batch"
End If

MsgBox "CheatGPT shortcut created on your Desktop!" & vbCrLf & vbCrLf & _
       "Type: " & strType & " Launcher" & vbCrLf & _
       "Double-click 'CheatGPT System' to start.", _
       vbInformation, "Success"
