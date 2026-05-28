/**
 * ============================================================
 *  FakeScope — Frontend Logic (main.js)
 * ============================================================
 *  Handles:
 *   - Text input & character counter
 *   - API call to POST /predict
 *   - Animated loading steps
 *   - Result display with confidence bar
 *   - Sample news injection
 *   - Error handling
 * ============================================================
 */

// ── DOM References ────────────────────────────────────────────
const newsInput        = document.getElementById('newsInput');
const charCount        = document.getElementById('charCount');
const analyzeBtn       = document.getElementById('analyzeBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultContainer  = document.getElementById('resultContainer');
const errorContainer   = document.getElementById('errorContainer');
const errorMsg         = document.getElementById('errorMsg');
const reanalyzeBtn     = document.getElementById('reanalyzeBtn');
const clearBtn         = document.getElementById('clearBtn');

// Result elements
const verdictBadge    = document.getElementById('verdictBadge');
const verdictIcon     = document.getElementById('verdictIcon');
const verdictText     = document.getElementById('verdictText');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill  = document.getElementById('confidenceFill');
const fakePct         = document.getElementById('fakePct');
const realPct         = document.getElementById('realPct');

// Loading step elements
const steps = [
  document.getElementById('step1'),
  document.getElementById('step2'),
  document.getElementById('step3'),
  document.getElementById('step4'),
];

// ── Sample News Articles ──────────────────────────────────────
const SAMPLES = {
  fake: `SHOCKING: Government Secretly Putting Mind-Control Chemicals in Tap Water, Whistleblower Claims

A former government employee has come forward with explosive allegations that authorities have been adding mind-altering substances to municipal water supplies across the country for decades. The whistleblower, who spoke anonymously, claims the program was designed to make citizens more compliant and less likely to question authority.

"The evidence is overwhelming," the source told an underground news blog. "Every major city is affected. They don't want you to know this."

Scientists and medical experts supposedly "hired by the government" have been silencing research that proves the contamination. Share this before it gets taken down. The mainstream media will NEVER report on this. Wake up, people — they are controlling your mind through your tap water every single day.`,

  real: `Federal Reserve Holds Interest Rates Steady, Signals Cautious Approach to Future Cuts

The Federal Reserve kept its benchmark interest rate unchanged Wednesday, maintaining the federal funds rate in its current target range as policymakers continue to assess inflation and labor market conditions.

Fed Chair Jerome Powell said at a press conference following the two-day meeting that officials want to see more evidence that inflation is sustainably moving toward the central bank's 2% target before making further adjustments.

"We're committed to returning inflation to our 2% goal," Powell said. "We believe our policy rate is well-positioned to address the risks we face."

The decision was unanimous among voting members of the Federal Open Market Committee. Economic data released this week showed the unemployment rate held steady while consumer spending remained resilient.`
};

// ── Character Counter ─────────────────────────────────────────
newsInput.addEventListener('input', () => {
  const len = newsInput.value.length;
  charCount.textContent = `${len.toLocaleString()} / 10,000`;

  // Color feedback
  if (len > 9000)       charCount.style.color = '#ff4d6d';
  else if (len > 7000)  charCount.style.color = '#ffaa44';
  else                  charCount.style.color = '';
});

// ── Sample Buttons ────────────────────────────────────────────
document.querySelectorAll('.sample-btn[data-type]').forEach(btn => {
  btn.addEventListener('click', () => {
    const type = btn.dataset.type;
    newsInput.value = SAMPLES[type];
    newsInput.dispatchEvent(new Event('input')); // update counter
    newsInput.focus();
    hideAll();
  });
});

clearBtn.addEventListener('click', () => {
  newsInput.value = '';
  newsInput.dispatchEvent(new Event('input'));
  hideAll();
  newsInput.focus();
});

// ── Re-analyze button ─────────────────────────────────────────
reanalyzeBtn.addEventListener('click', () => {
  newsInput.value = '';
  newsInput.dispatchEvent(new Event('input'));
  hideAll();
  newsInput.focus();
});

// ── Smooth nav scroll ─────────────────────────────────────────
document.querySelectorAll('.pill[href^="#"]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Update active pill
    document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    link.classList.add('active');
  });
});

// ── UI State Helpers ──────────────────────────────────────────
function hideAll() {
  loadingContainer.classList.remove('visible');
  resultContainer.classList.remove('visible');
  errorContainer.classList.remove('visible');
  // Reset loading steps
  if (steps && steps.length > 0) {
    steps.forEach(s => {
        s.classList.remove('active', 'done');
    });

    steps[0].classList.add('active');
}
}


function showLoading() {
  hideAll();
  analyzeBtn.disabled = true;
  loadingContainer.classList.add('visible');
  animateLoadingSteps();
}

function showError(message) {
  hideAll();
  analyzeBtn.disabled = false;
  errorMsg.textContent = message;
  errorContainer.classList.add('visible');
}

function showResult(data) {
  hideAll();
  analyzeBtn.disabled = false;

  const isReal = data.prediction === 'REAL';

  // Verdict badge
  verdictBadge.className = `verdict-badge ${isReal ? 'real' : 'fake'}`;
  verdictIcon.textContent = isReal ? '✓' : '✕';
  verdictText.textContent = data.prediction;

  // Confidence value
  confidenceValue.textContent = `${data.confidence}%`;

  // Confidence bar
  confidenceFill.className = `confidence-fill ${isReal ? 'real' : 'fake'}`;

  // Animate bar fill after small delay (allows CSS transition to fire)
  requestAnimationFrame(() => {
    setTimeout(() => {
      confidenceFill.style.width = `${data.confidence}%`;
    }, 80);
  });

  // Probability percentages
  fakePct.textContent = `${(data.fake_probability * 100).toFixed(1)}%`;
  realPct.textContent = `${(data.real_probability * 100).toFixed(1)}%`;

  resultContainer.classList.add('visible');
}

// ── Animated Loading Steps ────────────────────────────────────
function animateLoadingSteps() {
  const delays = [0, 400, 900, 1400]; // ms delays per step

  steps.forEach((step, i) => {
    setTimeout(() => {
      // Mark previous steps as done
      if (i > 0) steps[i - 1].classList.remove('active');
      if (i > 0) steps[i - 1].classList.add('done');
      step.classList.add('active');
    }, delays[i]);
  });
}

// ── Main Predict Function ─────────────────────────────────────
async function analyzeArticle() {
  const text = newsInput.value.trim();

  // Client-side validation
  if (!text) {
    showError('Please paste some news text before analyzing.');
    return;
  }
  if (text.length < 10) {
    showError('Text too short — please paste a full article or at least a paragraph.');
    return;
  }

  showLoading();

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    // Parse JSON regardless of status code (error messages come as JSON too)
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || `Server error (${response.status}). Please try again.`);
      return;
    }

    // Small delay so loading animation completes gracefully
    setTimeout(() => showResult(data), 1600);

  } catch (err) {
    console.error('Prediction error:', err);
    if (err instanceof TypeError && err.message.includes('fetch')) {
      showError('Cannot connect to server. Make sure Flask is running on localhost:5000.');
    } else {
      showError('Unexpected error. Please refresh the page and try again.');
    }
  }
}

// ── Event Listeners ───────────────────────────────────────────
analyzeBtn.addEventListener('click', analyzeArticle);

// Allow Ctrl+Enter to submit
newsInput.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    analyzeArticle();
  }
});

// ── Intersection Observer for nav active state ────────────────
const sections = document.querySelectorAll('section[id], main[id]');
const navLinks  = document.querySelectorAll('.pill[href^="#"]');

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      navLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
      });
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => observer.observe(s));

// ── Init ──────────────────────────────────────────────────────
hideAll();
console.log('%c FakeScope loaded ', 'background:#00e5a0;color:#080b0f;font-weight:bold;padding:4px 8px;border-radius:4px;');
analyzeBtn.addEventListener('click', analyzeArticle);

async function analyzeArticle() {

    const news = newsInput.value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: news })
    });

    const data = await response.json();

    result.innerText = data.prediction;
}
async function analyzeArticle() {

    const news = newsInput.value;

    if(news.trim() === "") {
        alert("Please enter news text");
        return;
    }

    try {

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: news
            })
        });

        const data = await response.json();

        document.getElementById("result").innerText =
            "Prediction: " + data.prediction;

    } catch(error) {
        console.log(error);
    }
}

analyzeBtn.addEventListener('click', analyzeArticle);