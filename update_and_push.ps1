# סקריפט להעלאת שינויים ל-GitHub

Write-Host "=== העלאת שינויים ל-GitHub ===" -ForegroundColor Cyan

# עבור לתיקיית הפרויקט
$projectDir = "C:\Users\moshe\OneDrive\שולחן העבודה\ConTech"
Set-Location $projectDir
Write-Host "עובד בתיקייה: $projectDir" -ForegroundColor Green

# בדוק סטטוס
Write-Host "`n1. בודק מה השתנה..." -ForegroundColor Cyan
git status

# שאל אם להמשיך
Write-Host "`nהאם להמשיך עם העלאה? (Y/N)" -ForegroundColor Yellow
$response = Read-Host

if ($response -ne "Y" -and $response -ne "y") {
    Write-Host "מבטל..." -ForegroundColor Red
    exit
}

# הוסף קבצים
Write-Host "`n2. מוסיף קבצים..." -ForegroundColor Cyan
git add .

# שאל על הודעת commit
Write-Host "`n3. הודעת commit:" -ForegroundColor Cyan
Write-Host "הכנס הודעה שמתארת את השינויים (או לחץ Enter לשימוש בהודעה ברירת מחדל):" -ForegroundColor Yellow
$commitMessage = Read-Host

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "עדכון: שיפורי UI ו-fixes"
}

# צור commit
Write-Host "`n4. יוצר commit..." -ForegroundColor Cyan
git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "שגיאה ביצירת commit" -ForegroundColor Red
    exit
}

# העלה ל-GitHub
Write-Host "`n5. מעלה ל-GitHub..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ הצלחה! השינויים הועלו ל-GitHub" -ForegroundColor Green
    Write-Host "`n📝 הערה: Streamlit Cloud יתעדכן אוטומטית תוך 2-3 דקות" -ForegroundColor Cyan
    Write-Host "בדוק ב: https://share.streamlit.io" -ForegroundColor Cyan
} else {
    Write-Host "`n⚠️ שגיאה בעת העלאה" -ForegroundColor Yellow
    Write-Host "ודא ש:" -ForegroundColor Gray
    Write-Host "  - אתה מחובר ל-GitHub" -ForegroundColor Gray
    Write-Host "  - יש לך הרשאות לכתוב ל-repository" -ForegroundColor Gray
    Write-Host "  - ה-remote מוגדר נכון (git remote -v)" -ForegroundColor Gray
}

Write-Host "`n=== סיום ===" -ForegroundColor Cyan
