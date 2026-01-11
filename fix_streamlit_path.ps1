# סקריפט לתיקון נתיב Streamlit Cloud
# מעתיק את app.py לתיקיית השורש של ה-repository

$repoRoot = "C:\Users\moshe\OneDrive\שולחן העבודה"
$sourceFile = "$repoRoot\ConTech\app.py"
$destFile = "$repoRoot\app.py"

# העתק את הקובץ
Copy-Item -Path $sourceFile -Destination $destFile -Force

Write-Host "✅ הקובץ הועתק לתיקיית השורש"

# עבור לתיקיית השורש והוסף ל-Git
Set-Location $repoRoot
git add app.py
git commit -m "Add app.py to root directory for Streamlit Cloud"
git push origin main

Write-Host "✅ הקובץ הועלה ל-GitHub"
