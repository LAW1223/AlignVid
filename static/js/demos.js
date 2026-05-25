// Datasets and comparisons manifest built from embedded data
// Each dataset includes an ordered list of items with: id, prompt, original, ours

const embeddedData = {
  Framepack: {
    files: [
      "1.mp4",
      "2.mp4",
      "3.mp4",
      "4.mp4",
      "5.mp4"
    ],
    prompts: [
      "Brighten the scene so that the person and background are clearly visible, and change the person's clothing to a vibrant red jacket",
      "The lush green mountain gradually erodes and disappears, leaving behind rolling sand dunes and a barren desert landscape",
      "A colorful Indian fighting kite (patang) with a long tail appears, tangled in the clothesline",
      "Flowers grew out of the steering wheel, and the picture became colorful",
      "The character's hair color gradually shift from red to white, transitioning smoothly over time"
    ]
  },
  Framepack_f1: {
    files: [
      "1.mp4",
      "2.mp4",
      "3.mp4",
      "4.mp4",
      "5.mp4"
    ],
    prompts: [
      "The sun and the clouds' colors shift gradually from red to yellow over time, while all other elements remain unchanged",
      "A thick fog bank rolls in over the surface of the dark river, limiting visibility",
      "A colorful Indian fighting kite (patang) with a long tail appears, tangled in the clothesline",
      "The character's hair color gradually shift from red to white, transitioning smoothly over time",
      "The man on horseback gradually transforms into a ghostly silhouette, his form slowly fading and dissolving into translucent light, while his horse begins to fade into the serene landscape"
    ]
  },
  "Wan2.1": {
    files: [
      "1.mp4",
      "2.mp4",
      "3.mp4",
      "4.mp4",
      "5.mp4"
    ],
    prompts: [
      "The gray convertible car move forward and gradually drives out of sight",
      "Trees sprout and quickly grow across the mountains, covering the rocky slopes with lush green foliage",
      "A young tree sprouts at the front of the house and grows quickly until it stands full-height in front of the facade",
      "The man extending his arm forward and releasing the diamond from his fingers",
      "The character's hair color gradually shift from red to white, transitioning smoothly over time"
    ]
  }
};

const demoDatasets = {
  Framepack: [],
  Framepack_f1: [],
  "Wan2.1": []
};

function buildDatasetItems(datasetName, prompts, fileNames) {
  const sorted = [...fileNames].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const items = [];
  const maxLen = Math.min(prompts.length, sorted.length);
  for (let i = 0; i < maxLen; i++) {
    const fname = sorted[i];
    const id = fname.replace(/\.mp4$/i, "");
    const cacheBust = `?v=${encodeURIComponent(fname)}`;
    items.push({
      id,
      prompt: prompts[i],
      original: `./demo_video/${datasetName}/original/${fname}${cacheBust}`,
      ours: `./demo_video/${datasetName}/ours/${fname}${cacheBust}`
    });
  }
  return items;
}

function initDemoDatasetsEmbedded() {
  const datasets = Object.keys(embeddedData);
  for (const name of datasets) {
    const { prompts, files } = embeddedData[name];
    demoDatasets[name] = buildDatasetItems(name, prompts, files);
  }
}

function createVideo(src) {
  const video = document.createElement('video');
  video.controls = true;
  video.muted = true; // allow autoplay on most browsers
  video.autoplay = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = 'metadata';
  video.width = 512;

  const source = document.createElement('source');
  source.src = src;
  source.type = 'video/mp4';
  source.onerror = function() {
    // Try absolute-from-root as fallback
    const url = new URL(src, location.origin).toString();
    if (source.src !== url) {
      source.src = url;
      video.load();
    }
  };

  video.appendChild(source);
  video.appendChild(document.createTextNode('Your browser does not support the video tag.'));
  return video;
}

function renderComparisonList(container, items) {
  items.forEach((item, index) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'box';

    const header = document.createElement('h3');
    header.className = 'title is-5';
    header.textContent = `Example ${String(index + 1).padStart(2,'0')}`;

    const prompt = document.createElement('p');
    prompt.className = 'content';
    prompt.textContent = item.prompt;

    const row = document.createElement('div');
    row.className = 'columns is-vcentered is-multiline';

    const left = document.createElement('div');
    left.className = 'column is-6 has-text-centered';
    const leftTitle = document.createElement('p');
    leftTitle.className = 'has-text-weight-semibold';
    leftTitle.textContent = 'Original';
    const v1 = createVideo(item.original);
    left.appendChild(leftTitle);
    left.appendChild(v1);

    const right = document.createElement('div');
    right.className = 'column is-6 has-text-centered';
    const rightTitle = document.createElement('p');
    rightTitle.className = 'has-text-weight-semibold';
    rightTitle.textContent = 'Ours';
    const v2 = createVideo(item.ours);
    right.appendChild(rightTitle);
    right.appendChild(v2);

    row.appendChild(left);
    row.appendChild(right);

    wrapper.appendChild(header);
    wrapper.appendChild(prompt);
    wrapper.appendChild(row);

    container.appendChild(wrapper);
  });
}

function setupVideoComparisons() {
  const host = document.getElementById('video-comparisons');
  if (!host) return;

  // Build datasets from embedded data (no fetch, works on file:// and static hosts)
  initDemoDatasetsEmbedded();

  // Render all datasets without selector
  const datasets = Object.keys(demoDatasets);
  datasets.forEach((name) => {
    const section = document.createElement('section');
    section.className = 'section';
    const container = document.createElement('div');
    container.className = 'container is-max-desktop';

    const title = document.createElement('h3');
    title.className = 'title is-4';
    title.textContent = name;

    const list = document.createElement('div');
    renderComparisonList(list, demoDatasets[name]);

    container.appendChild(title);
    container.appendChild(list);
    section.appendChild(container);
    host.appendChild(section);
  });
}

// Auto-init on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupVideoComparisons);
} else {
  setupVideoComparisons();
} 