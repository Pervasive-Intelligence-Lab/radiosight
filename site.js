// Copy the BibTeX entry; the button says "Copied" for two seconds.
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('copy-bib');
  const bib = document.getElementById('bibtex');
  if (!btn || !bib || !navigator.clipboard) return;
  btn.addEventListener('click', () => {
    navigator.clipboard.writeText(bib.textContent).then(() => {
      btn.textContent = 'Copied';
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });
  });
});
