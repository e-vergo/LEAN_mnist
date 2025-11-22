# LEAN_mnist Landing Page Deployment Checklist

**Quick reference for deploying the landing page to GitHub Pages**

---

## Pre-Deployment Checklist

### 1. Update GitHub Username (REQUIRED)
Before deploying, replace the placeholder username with your actual GitHub username:

```bash
cd /Users/eric/LEAN_mnist
sed -i '' 's/yourusername/YOUR_GITHUB_USERNAME/g' docs/index.html
```

**Files affected:**
- CTA button links (View Proofs)
- Footer links (GitHub Repository, Issues)
- Open Graph meta tags
- External resource links

**Verification:**
```bash
grep -n "yourusername" docs/index.html
# Should return 0 results after replacement
```

### 2. Local Preview Test
Test the page locally before deploying:

```bash
cd docs
python3 -m http.server 8000
# Visit: http://localhost:8000
```

**Test checklist:**
- [ ] All sections render correctly
- [ ] Visual showcase displays (ASCII art, diagram, code)
- [ ] Training chart displays
- [ ] All links work (except parent directory links)
- [ ] Responsive design works (resize browser)
- [ ] No console errors (open DevTools → Console)

### 3. File Verification
Ensure all required files exist:

```bash
ls -lh docs/
# Should show:
# - index.html (~38 KB)
# - styles.css (~21 KB)
# - content.md (~12 KB)
# - README.md (~19 KB)
# - assets/ (directory)
```

---

## GitHub Pages Setup (First-Time Only)

### Step 1: Enable GitHub Pages
1. Go to your GitHub repository
2. Click **Settings** (top navigation)
3. Scroll down to **Pages** (left sidebar under "Code and automation")

### Step 2: Configure Source
In the Pages settings:
- **Source:** Deploy from a branch
- **Branch:** main (or master, depending on your default branch)
- **Folder:** /docs
- Click **Save**

### Step 3: Wait for Deployment
- Initial deployment takes 1-10 minutes
- Check **Actions** tab for progress
- Look for "pages build and deployment" workflow

### Step 4: Verify URL
Your site will be available at:
```
https://YOUR_USERNAME.github.io/LEAN_mnist/
```

---

## Deployment Process (After Setup)

### Push Changes to Trigger Deployment
```bash
cd /Users/eric/LEAN_mnist
git add docs/
git commit -m "Deploy landing page to GitHub Pages"
git push origin main
```

### Monitor Deployment
1. Go to repository → **Actions** tab
2. Look for most recent "pages build and deployment" run
3. Green checkmark = success, red X = failure
4. Click on run to see deployment logs

### Typical Timeline
- **Commit pushed:** Immediate
- **Workflow triggered:** < 30 seconds
- **Build and deploy:** 30-90 seconds
- **Site updated:** < 2 minutes total

---

## Post-Deployment Verification

### 1. Check Site Loads
Visit `https://YOUR_USERNAME.github.io/LEAN_mnist/`

**Expected result:**
- Page loads successfully (no 404)
- Styling appears correctly
- No broken images

### 2. Test All Sections
Scroll through entire page and verify:
- [ ] Hero section displays with CTA buttons
- [ ] Visual showcase (3 boxes) renders correctly
- [ ] Main theorem box displays
- [ ] Achievement cards (4 cards) show up
- [ ] Training chart appears
- [ ] Module reference list is readable
- [ ] Proof strategy section loads
- [ ] Getting started section displays
- [ ] Footer renders with all sections

### 3. Test Links
Click each link type:
- [ ] Documentation links (open parent directory files)
- [ ] GitHub links (go to repository)
- [ ] External links (Lean docs, SciLean, Zulip)
- [ ] Internal anchors (#architecture)

### 4. Test Responsive Design
Resize browser or use DevTools:
- [ ] Desktop (> 1024px): 3-column showcase, 2-column grid
- [ ] Tablet (768-1024px): Responsive layout
- [ ] Mobile (< 768px): Single-column stack

### 5. Test Accessibility
- [ ] Skip link works (press Tab, Enter)
- [ ] Keyboard navigation works (Tab through page)
- [ ] Links have visible focus indicators
- [ ] Headings are in logical order (h1 → h2 → h3)

---

## Troubleshooting Common Issues

### Issue: 404 Page Not Found
**Cause:** GitHub Pages not configured correctly
**Solution:**
1. Check Settings → Pages → Source is set to "main" and "/docs"
2. Verify index.html exists in docs/ folder
3. Wait up to 10 minutes for first deployment

### Issue: Page Loads but No Styling
**Cause:** CSS file not found
**Solution:**
1. Check `docs/styles.css` exists
2. Verify `<link rel="stylesheet" href="styles.css">` in index.html
3. Hard refresh browser (Ctrl+Shift+R)

### Issue: Broken Links to Documentation
**Cause:** Links pointing to wrong directory
**Solution:**
- Links to parent files should use `../filename.md`
- Links to same directory use `./filename.md` or `filename.md`
- Verify: `grep 'href="\./[A-Z]' docs/index.html` should return 0 results

### Issue: SVGs Not Displaying
**Cause:** SVG syntax error or encoding issue
**Solution:**
1. Validate SVG: `xmllint --noout docs/assets/*.svg`
2. Check browser console for errors
3. Verify SVG is properly inlined in HTML

### Issue: Mobile Layout Broken
**Cause:** CSS media queries not working
**Solution:**
1. Check viewport meta tag: `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
2. Test in browser DevTools device emulator
3. Verify media queries in styles.css

---

## Updating Content After Deployment

### Update Statistics
```bash
# 1. Edit docs/index.html
# Find and replace statistics:
grep -n "93%" docs/index.html  # Test accuracy
grep -n ">26<" docs/index.html  # Theorem count
grep -n ">4<" docs/index.html   # Sorry count
grep -n ">9<" docs/index.html   # Axiom count

# 2. Commit and push
git add docs/index.html
git commit -m "Update landing page statistics"
git push origin main
```

### Update Training Chart
```bash
# 1. Update training data
vim docs/assets/training-data.csv

# 2. Regenerate SVG (see docs/assets/README.md)
python3 scripts/generate_training_curve.py

# 3. Copy SVG into index.html (lines 397-590)
# OR keep as inline SVG (current approach)

# 4. Commit and push
git add docs/
git commit -m "Update training convergence chart"
git push origin main
```

### Add New Section
```bash
# 1. Plan content
vim docs/content.md  # Add new section content

# 2. Add HTML structure
vim docs/index.html  # Insert new <section>

# 3. Add CSS styles
vim docs/styles.css  # Add new classes

# 4. Test locally
cd docs && python3 -m http.server 8000

# 5. Commit and push
git add docs/
git commit -m "Add new section: [section name]"
git push origin main
```

---

## Performance Optimization Tips

### Monitor Page Size
```bash
wc -c docs/index.html docs/styles.css
# Target: < 60 KB total
```

### Check Load Performance
Use Lighthouse in Chrome DevTools:
1. Open page in Chrome
2. F12 → Lighthouse tab
3. Generate report
4. Target scores: Performance 90+, Accessibility 95+

### Optimize Assets
- Keep SVGs optimized (use SVGO)
- Inline critical assets (current approach)
- Avoid external dependencies
- Use system fonts (no web fonts)

---

## Rollback Procedure

### If Deployment Fails
```bash
# 1. Revert to previous commit
git log --oneline docs/  # Find last good commit
git revert COMMIT_HASH

# 2. Push rollback
git push origin main

# 3. Verify site restored
# Visit: https://YOUR_USERNAME.github.io/LEAN_mnist/
```

### Emergency Fix
```bash
# Quick fix without full commit history
# 1. Fix the file
vim docs/index.html

# 2. Commit with descriptive message
git add docs/index.html
git commit -m "HOTFIX: [description of fix]"
git push origin main

# 3. Deployment auto-triggers
```

---

## Maintenance Schedule

### Weekly
- [ ] Verify site is online and accessible
- [ ] Check for any broken links (use link checker tool)
- [ ] Review GitHub Issues for bug reports

### After Each Training Run
- [ ] Update accuracy statistics
- [ ] Update training chart with new data
- [ ] Update best model epoch number

### After Verification Work
- [ ] Update theorem count
- [ ] Update sorry count
- [ ] Update axiom count (if changed)

### Quarterly
- [ ] Review content for accuracy
- [ ] Update external links (check for dead links)
- [ ] Test on latest browser versions
- [ ] Run Lighthouse audit for performance

---

## Contact & Support

### Deployment Issues
**GitHub Pages Status:** https://www.githubstatus.com/
**Documentation:** https://docs.github.com/en/pages

### Questions
- **Project Issues:** https://github.com/yourusername/LEAN_mnist/issues
- **Lean Community:** https://leanprover.zulipchat.com/#narrow/stream/113489-new-members

---

## Quick Reference Commands

```bash
# Local preview
cd docs && python3 -m http.server 8000

# Check file sizes
wc -c docs/index.html docs/styles.css

# Find and replace GitHub username
sed -i '' 's/yourusername/ACTUAL_USERNAME/g' docs/index.html

# Verify no placeholder usernames remain
grep -n "yourusername" docs/index.html

# Commit and deploy
git add docs/
git commit -m "Update landing page"
git push origin main

# Check deployment status
# Visit: https://github.com/yourusername/LEAN_mnist/actions
```

---

**Last Updated:** November 21, 2025
**Version:** 1.0
**Status:** Ready for deployment ✅
