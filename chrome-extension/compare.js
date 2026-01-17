/**
 * Property Comparison Page
 * Fetches similar listings from API and renders comparison view
 */

const API_BASE = 'https://dashboard-fawn-nu-59.vercel.app';

// Parse URL params
function getParams() {
  const params = new URLSearchParams(window.location.search);
  return {
    address: params.get('address') || 'Unknown Property',
    postcode: params.get('postcode') || '',
    beds: parseInt(params.get('beds'), 10) || 2,
    sqft: parseInt(params.get('sqft'), 10) || 0,
    price: parseInt(params.get('price'), 10) || 0,
    fairValue: parseInt(params.get('fairValue'), 10) || 0,
    type: params.get('type') || '',
    url: params.get('url') || '',
    propertyId: params.get('propertyId') || '',
  };
}

// Format currency
function formatPrice(price) {
  return '£' + price.toLocaleString();
}

// Format percentage difference
function formatDiff(asking, fair) {
  if (!fair) return '';
  const diff = ((asking - fair) / fair) * 100;
  const sign = diff > 0 ? '+' : '';
  const cls = diff > 5 ? 'overpriced' : diff < -5 ? 'underpriced' : 'fair';
  return `<span class="${cls}">${sign}${diff.toFixed(0)}%</span>`;
}

// Render your property section
function renderYourProperty(params) {
  document.getElementById('your-address').textContent = params.address;
  document.getElementById('your-beds').textContent = params.beds;
  document.getElementById('your-sqft').textContent = params.sqft > 0 ? params.sqft.toLocaleString() : 'N/A';
  document.getElementById('your-postcode').textContent = params.postcode;
  document.getElementById('your-price').textContent = formatPrice(params.price) + ' pcm';

  if (params.sqft > 0) {
    const ppsf = (params.price / params.sqft).toFixed(2);
    document.getElementById('your-ppsf').textContent = '£' + ppsf;
  } else {
    document.getElementById('your-ppsf').textContent = 'N/A';
  }

  if (params.fairValue > 0) {
    document.getElementById('your-fair-value').textContent = formatPrice(params.fairValue);
    document.getElementById('your-diff').innerHTML = formatDiff(params.price, params.fairValue);
  } else {
    document.getElementById('your-fair-value').textContent = 'N/A';
    document.getElementById('your-diff').textContent = '';
  }

  // Back link
  if (params.url) {
    document.getElementById('back-link').href = params.url;
  }
}

// Render stats summary
function renderStats(stats, yourPrice) {
  document.getElementById('stat-peers').textContent = stats.peer_count;
  document.getElementById('stat-avg-price').textContent = formatPrice(stats.avg_price);
  document.getElementById('stat-avg-ppsf').textContent = stats.avg_ppsf ? '£' + stats.avg_ppsf.toFixed(2) : 'N/A';
  document.getElementById('stat-percentile').textContent = stats.your_percentile + '%';
}

// Render price positioning chart
function renderPriceRange(stats, yourPrice) {
  const { min_price, max_price, avg_price } = stats;
  const range = max_price - min_price || 1;

  document.getElementById('range-min').textContent = formatPrice(min_price);
  document.getElementById('range-max').textContent = formatPrice(max_price);

  // Position markers
  const yourPos = Math.max(0, Math.min(100, ((yourPrice - min_price) / range) * 100));
  const avgPos = Math.max(0, Math.min(100, ((avg_price - min_price) / range) * 100));

  document.getElementById('your-marker').style.left = yourPos + '%';
  document.getElementById('avg-marker').style.left = avgPos + '%';

  // Fill bar from min to your position
  document.getElementById('range-fill').style.width = yourPos + '%';
}

// Render peer cards
function renderPeers(peers) {
  const grid = document.getElementById('peers-grid');
  const noPeers = document.getElementById('no-peers');

  if (peers.length === 0) {
    grid.classList.add('hidden');
    noPeers.classList.remove('hidden');
    return;
  }

  document.getElementById('peers-count').textContent = `(${peers.length})`;

  grid.innerHTML = peers.map(peer => {
    const ppsf = peer.ppsf ? `£${peer.ppsf.toFixed(2)}/sqft` : '';
    const similarity = Math.round(peer.similarity_score * 100);
    const sourceClass = ['savills', 'knightfrank'].includes(peer.source) ? 'premium' : '';

    return `
      <a href="${peer.url}" target="_blank" class="peer-card ${sourceClass}">
        <div class="peer-header">
          <span class="peer-source">${peer.source}</span>
          <span class="peer-match">${similarity}% match</span>
        </div>
        <div class="peer-address">${peer.address}</div>
        <div class="peer-meta">
          ${peer.bedrooms} bed
          ${peer.size_sqft ? ` · ${peer.size_sqft.toLocaleString()} sqft` : ''}
        </div>
        <div class="peer-price">
          <span class="peer-price-main">${formatPrice(peer.price_pcm)} pcm</span>
          ${ppsf ? `<span class="peer-ppsf">${ppsf}</span>` : ''}
        </div>
      </a>
    `;
  }).join('');
}

// Generate market insights
function renderInsights(stats, yourPrice, yourSqft, peers) {
  const insights = [];
  const { avg_price, avg_ppsf, your_percentile, peer_count } = stats;

  // Price vs average
  const priceDiff = ((yourPrice - avg_price) / avg_price) * 100;
  if (priceDiff > 10) {
    insights.push(`This property is priced <strong>${priceDiff.toFixed(0)}% above</strong> the peer average of ${formatPrice(avg_price)}`);
  } else if (priceDiff < -10) {
    insights.push(`This property is priced <strong>${Math.abs(priceDiff).toFixed(0)}% below</strong> the peer average - potential value`);
  } else {
    insights.push(`This property is priced <strong>in line</strong> with similar properties in the area`);
  }

  // Percentile
  if (your_percentile >= 75) {
    insights.push(`At the <strong>${your_percentile}th percentile</strong> - one of the more expensive options among peers`);
  } else if (your_percentile <= 25) {
    insights.push(`At the <strong>${your_percentile}th percentile</strong> - one of the more affordable options`);
  }

  // Better value options
  const cheaperBetter = peers.filter(p =>
    p.price_pcm < yourPrice &&
    (!yourSqft || !p.size_sqft || p.size_sqft >= yourSqft * 0.9)
  );
  if (cheaperBetter.length > 0) {
    insights.push(`<strong>${cheaperBetter.length}</strong> similar properties are available at a lower price`);
  }

  // Premium agents
  const premiumCount = peers.filter(p => ['savills', 'knightfrank'].includes(p.source)).length;
  if (premiumCount > 0) {
    insights.push(`<strong>${premiumCount}</strong> listings from premium agents (Savills, Knight Frank) - often better data quality`);
  }

  document.getElementById('insights-list').innerHTML = insights.map(i => `<li>${i}</li>`).join('');
}

// Main function
async function init() {
  const params = getParams();

  // Render your property immediately
  renderYourProperty(params);

  // Validate required params
  if (!params.postcode || !params.price) {
    showError('Missing required property data');
    return;
  }

  try {
    // Fetch similar listings
    const queryParams = new URLSearchParams({
      postcode: params.postcode,
      beds: params.beds.toString(),
      price: params.price.toString(),
    });
    if (params.sqft) queryParams.set('sqft', params.sqft.toString());
    if (params.type) queryParams.set('type', params.type);
    if (params.propertyId) queryParams.set('exclude', params.propertyId);

    const response = await fetch(`${API_BASE}/api/similar?${queryParams}`);

    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }

    const data = await response.json();
    console.log('[Compare] API response:', data);

    // Render results
    renderStats(data.stats, params.price);
    renderPriceRange(data.stats, params.price);
    renderPeers(data.peers);
    renderInsights(data.stats, params.price, params.sqft, data.peers);

    // Show content
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('content').classList.remove('hidden');

  } catch (error) {
    console.error('[Compare] Error:', error);
    showError(error.message);
  }
}

function showError(message) {
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('error').classList.remove('hidden');
  document.getElementById('error-message').textContent = message;
}

// Start
document.addEventListener('DOMContentLoaded', init);
