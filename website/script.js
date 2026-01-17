/* ==========================================================================
   Rent Fair Value - Landing Page Scripts
   ========================================================================== */

(function() {
  'use strict';

  /* Utility: Throttle function
     ======================================================================== */
  function throttle(func, limit) {
    let inThrottle;
    return function() {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  /* Header scroll effect
     ======================================================================== */
  function initHeader() {
    const header = document.getElementById('header');
    if (!header) return;

    let lastScrolled = null;

    function updateHeader() {
      const isScrolled = window.scrollY > 10;
      // Only update DOM if state changed
      if (isScrolled !== lastScrolled) {
        header.classList.toggle('scrolled', isScrolled);
        lastScrolled = isScrolled;
      }
    }

    window.addEventListener('scroll', throttle(updateHeader, 100), { passive: true });
    updateHeader();
  }

  /* FAQ Accordion
     ======================================================================== */
  function initFAQ() {
    const faqItems = document.querySelectorAll('.faq-item');
    if (!faqItems.length) return;

    faqItems.forEach((item, index) => {
      const question = item.querySelector('.faq-item__question');
      const answer = item.querySelector('.faq-item__answer');
      if (!question || !answer) return;

      // Set up ARIA attributes
      const answerId = `faq-answer-${index}`;
      answer.id = answerId;
      question.setAttribute('aria-controls', answerId);

      question.addEventListener('click', () => {
        const isOpen = item.classList.contains('open');

        // Close all items (accordion behavior)
        faqItems.forEach(i => {
          const btn = i.querySelector('.faq-item__question');
          const ans = i.querySelector('.faq-item__answer');
          i.classList.remove('open');
          if (btn) btn.setAttribute('aria-expanded', 'false');
          if (ans) ans.setAttribute('aria-hidden', 'true');
        });

        // Toggle current item
        if (!isOpen) {
          item.classList.add('open');
          question.setAttribute('aria-expanded', 'true');
          answer.setAttribute('aria-hidden', 'false');

          // Track FAQ expansion
          if (typeof posthog !== 'undefined') {
            const questionText = question.querySelector('span')?.textContent || '';
            posthog.capture('faq_expand', { question: questionText });
          }
        }
      });

      // Initialize ARIA state
      question.setAttribute('aria-expanded', 'false');
      answer.setAttribute('aria-hidden', 'true');
    });
  }

  /* CTA Click Tracking
     ======================================================================== */
  function initCTATracking() {
    const ctaButtons = document.querySelectorAll('[data-cta]');
    if (!ctaButtons.length) return;

    ctaButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const location = btn.getAttribute('data-cta');
        const text = btn.textContent.trim();

        // Track click
        if (typeof posthog !== 'undefined') {
          posthog.capture('cta_click', {
            button_location: location,
            button_text: text
          });
        }

        // For placeholder links, prevent navigation
        // TODO: Replace # with actual Chrome Web Store URL before launch
        const href = btn.getAttribute('href');
        if (href === '#') {
          e.preventDefault();
        }
      });
    });
  }

  /* Scroll Depth Tracking
     ======================================================================== */
  function initScrollTracking() {
    const thresholds = [25, 50, 75, 100];
    const triggered = new Set();
    let scrollListener;

    function checkScroll() {
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (scrollHeight <= 0) return;

      const scrollPercent = Math.round((window.scrollY / scrollHeight) * 100);

      thresholds.forEach(threshold => {
        if (scrollPercent >= threshold && !triggered.has(threshold)) {
          triggered.add(threshold);

          if (typeof posthog !== 'undefined') {
            posthog.capture('scroll_depth', { depth: threshold });
          }

          // Remove listener once all thresholds triggered
          if (triggered.size === thresholds.length && scrollListener) {
            window.removeEventListener('scroll', scrollListener);
          }
        }
      });
    }

    scrollListener = throttle(checkScroll, 200);
    window.addEventListener('scroll', scrollListener, { passive: true });
  }

  /* Smooth scroll for anchor links
     ======================================================================== */
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', (e) => {
        const href = anchor.getAttribute('href');
        // Skip placeholder links
        if (href === '#') {
          e.preventDefault();
          return;
        }

        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  }

  /* Initialize all
     ======================================================================== */
  function init() {
    initHeader();
    initFAQ();
    initCTATracking();
    initScrollTracking();
    initSmoothScroll();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
