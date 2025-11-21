# LEAN_mnist Landing Page

Static landing page for the LEAN_mnist verified neural network project.

## Structure

- `index.html` - Main landing page (37.6 KB)
- `styles.css` - Stylesheet with ChebyshevCircles.io aesthetic (20.5 KB)
- `content.md` - Source content for reference (12 KB)
- `assets/` - Visual assets (SVGs, code snippets, ASCII art)

**Total page size:** ~58 KB (HTML + CSS + inlined assets)

## Updating Content

### Change Text Content
Edit `content.md` for content planning, then manually update the corresponding sections in `index.html`.

### Update Statistics
Search for these values in `index.html` and update as needed:
- `93%` - Test accuracy (appears in hero, badges, achievement cards, training chart)
- `26` - Proven theorems (theorem count badge)
- `4` - Remaining sorries (sorry count badge)
- `9` - Axioms (axiom count badge)
- `60,000` - Training samples
- `3.3 hours` - Training time

### Update Visual Assets
See `assets/README.md` for instructions on regenerating:
- `training-curve.svg` - Training convergence chart (from logs)
- `architecture.svg` - Network architecture diagram (manual SVG editing)
- Code snippets in `assets/*.txt` - Extract from source files

## GitHub Pages Deployment

### First-Time Setup
1. Go to repository Settings
2. Navigate to Pages section
3. Source: **Deploy from a branch**
4. Branch: **main** (or **master**)
5. Folder: **/docs**
6. Click Save

### Deployment URL
Once configured, the site will be available at:
```
https://YOUR_USERNAME.github.io/LEAN_mnist/
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Automatic Updates
Any changes pushed to the `main` branch in the `docs/` folder will automatically deploy within 1-2 minutes.

### Deployment Status
Check deployment status:
1. Go to repository **Actions** tab
2. Look for "pages build and deployment" workflow
3. Green checkmark = successful deployment
4. Red X = deployment failed (check logs)

## Local Preview

### Using Python
```bash
cd docs
python3 -m http.server 8000
# Visit: http://localhost:8000
```

### Using Node.js
```bash
cd docs
npx http-server
# Visit: http://localhost:8080
```

### Direct File Open
Simply open `docs/index.html` in your browser. Note: Some features may not work without a local server (e.g., CORS for external assets).

## Maintenance

### Adding New Sections
1. **Plan content:** Write content in `content.md`
2. **Update HTML:** Add HTML structure to `index.html`
3. **Style it:** Add CSS styles to `styles.css`
4. **Test responsive design:** Check on mobile, tablet, desktop

### Regenerating Assets
Refer to `assets/README.md` for detailed instructions on:
- Creating training curve SVGs from log files
- Editing the architecture diagram
- Extracting code snippets from source files
- Updating ASCII art renderings

### Color Palette
Consistent with ChebyshevCircles.io aesthetic:
- **Background:** `#f5f5f5` (light gray)
- **Content background:** `#ffffff` (white)
- **Accent:** `#3498db` (blue)
- **Headings:** `#2c3e50` (dark blue-gray)
- **Body text:** `#555555` (medium gray)
- **Success:** `#27ae60` (green)

### Typography
- **System fonts:** -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial
- **Monospace:** SF Mono, Monaco, Cascadia Code, Courier New
- **Base size:** 16px (1rem)
- **Line height:** 1.6 (body), 1.2 (headings)

## Browser Support

### Supported Browsers
- **Chrome/Edge:** Latest 2 versions (Chromium-based)
- **Firefox:** Latest 2 versions
- **Safari:** Latest 2 versions (macOS, iOS)
- **Mobile browsers:** iOS Safari, Chrome Android

### CSS Features Used
All features have excellent browser support (95%+ coverage):
- CSS Grid Layout
- CSS Custom Properties (variables)
- Flexbox
- Media Queries
- Sticky positioning
- SVG inline embedding

### Testing Checklist
- [ ] Desktop Chrome (1920×1080)
- [ ] Desktop Firefox (1920×1080)
- [ ] Desktop Safari (macOS)
- [ ] Tablet landscape (1024×768)
- [ ] Tablet portrait (768×1024)
- [ ] Mobile landscape (667×375)
- [ ] Mobile portrait (375×667)

## Performance

### Target Metrics
- **Page size:** < 60 KB (currently ~58 KB)
- **Load time:** < 3 seconds on 3G
- **Lighthouse score:** 95+ (Performance, Accessibility, Best Practices, SEO)

### Optimization Techniques
- Inline SVGs (no external requests)
- System fonts (no web font downloads)
- Minimal CSS (20 KB, no frameworks)
- No JavaScript dependencies
- Optimized images (SVGs are vector-based)

### Performance Testing
```bash
# Using Lighthouse CLI
npm install -g lighthouse
lighthouse http://localhost:8000 --view

# Check file sizes
du -h index.html styles.css
```

## Accessibility

### Features Implemented
- **Skip link:** Jump to main content (keyboard navigation)
- **Semantic HTML:** Proper heading hierarchy (h1→h2→h3)
- **ARIA labels:** Navigation landmarks, roles
- **Alt text:** All images and SVGs have descriptive alt text
- **Color contrast:** WCAG AA compliant (4.5:1 minimum)
- **Keyboard navigation:** All interactive elements accessible via Tab
- **Focus indicators:** Visible focus outlines on interactive elements

### Testing Accessibility
- **Lighthouse:** Accessibility score should be 95+
- **Screen reader:** Test with VoiceOver (macOS) or NVDA (Windows)
- **Keyboard only:** Navigate entire page using only keyboard
- **Color blindness:** Use Color Oracle to simulate color vision deficiencies

## Responsive Design

### Breakpoints
- **Desktop:** > 1024px (default styles)
- **Tablet:** 768px - 1024px (2-column layouts become responsive)
- **Mobile:** < 768px (single-column stacking)
- **Small mobile:** < 480px (reduced font sizes)

### Layout Changes
- **Desktop:** 3-column visual showcase, 2-column achievement grid
- **Tablet:** 2-column visual showcase (last item spans full width), 2-column grid
- **Mobile:** Single-column everything, stacked CTA buttons, simplified navigation

### Testing Responsive Design
Use browser DevTools:
1. Open DevTools (F12)
2. Toggle device toolbar (Ctrl+Shift+M)
3. Test common devices: iPhone 12, iPad, Desktop
4. Check both portrait and landscape orientations

## File Organization

```
docs/
├── index.html          # Main landing page
├── styles.css          # Complete stylesheet
├── content.md          # Content source (reference only)
├── README.md           # This file
└── assets/             # Visual assets
    ├── architecture.svg         # Network diagram
    ├── training-curve.svg       # Training convergence chart
    ├── mnist-ascii-clean.txt    # ASCII art rendering
    ├── manual-backprop-code.txt # Code snippet
    ├── training-data.csv        # Training curve data
    ├── README.md                # Asset documentation
    ├── ASSET_INVENTORY.txt      # Asset manifest
    └── COMPLETION_REPORT.md     # Agent 2 completion report
```

## Common Tasks

### Update Training Results
1. Run training: `lake exe mnistTrainFull`
2. Extract accuracy data from logs
3. Update `assets/training-data.csv`
4. Regenerate `assets/training-curve.svg`
5. Update statistics in `index.html`

### Fix Broken Links
Search for broken links:
```bash
grep -n 'href="' docs/index.html | grep -v "http"
```

Update paths as needed (relative to `docs/` directory):
- Documentation files: `../GETTING_STARTED.md` (parent directory)
- GitHub links: `https://github.com/yourusername/LEAN_mnist/...`

### Add New Achievement Card
1. Edit `index.html` around line 344 (achievement-grid section)
2. Copy existing card structure:
```html
<article class="achievement-card">
  <div class="card-stat">VALUE</div>
  <h3>Title</h3>
  <p>Description text...</p>
</article>
```
3. Adjust grid if needed (currently 2×2 layout)

### Customize Colors
Edit CSS variables in `styles.css` (lines 9-23):
```css
:root {
  --background: #f5f5f5;
  --accent: #3498db;        /* Change blue accent */
  --heading: #2c3e50;       /* Change heading color */
  /* ... */
}
```

## Troubleshooting

### Issue: CSS Not Updating
**Solution:** Hard refresh the page
- **Chrome/Firefox:** Ctrl+Shift+R (Windows/Linux), Cmd+Shift+R (Mac)
- **Safari:** Cmd+Option+R

### Issue: Links Not Working on GitHub Pages
**Solution:** Check relative paths
- Files in parent directory: Use `../filename.md`
- Files in same directory: Use `./filename.md` or `filename.md`
- External links: Use full URL with `https://`

### Issue: SVGs Not Displaying
**Solution:** Verify SVG syntax
```bash
# Validate SVG
xmllint --noout docs/assets/architecture.svg
```

### Issue: Page Looks Different Locally vs GitHub Pages
**Solution:** Check base URL
- Local: `file:///path/to/docs/index.html`
- GitHub Pages: `https://username.github.io/LEAN_mnist/`
- Ensure all paths work in both contexts

## Version History

### v1.0 (November 21, 2025)
- Initial landing page release
- Complete HTML structure with semantic markup
- CSS styling matching ChebyshevCircles.io aesthetic
- Triple visual showcase (ASCII art, architecture, code)
- Training convergence chart (93% accuracy)
- Responsive design (desktop, tablet, mobile)
- Accessibility features (skip link, ARIA labels)
- GitHub Pages ready

### Future Enhancements
- [ ] Add favicon
- [ ] Add GitHub stars/forks badges
- [ ] Minify CSS for production (keep readable version)
- [ ] Add meta tags for Twitter Cards
- [ ] Create dark mode variant
- [ ] Add smooth scroll animations
- [ ] Include video demo (optional)

## License

Same as the project (see LICENSE file in repository root).

## Contact

**Issues:** [GitHub Issues](https://github.com/yourusername/LEAN_mnist/issues)
**Community:** [Lean Zulip #scientific-computing](https://leanprover.zulipchat.com/)

---

**Last Updated:** November 21, 2025
**Maintained by:** LEAN_mnist project contributors
