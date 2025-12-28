let map;
let markers = [];
let mapProvider = null;
let currentMode = 'location';
let barChart = null;
let pieChart = null;

document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', function() {
            switchMode(this.dataset.mode);
        });
    });

    document.getElementById('search-location-btn').addEventListener('click', handleLocationSearch);
    document.getElementById('analyze-csv-btn').addEventListener('click', handleCSVAnalysis);
    document.getElementById('use-auto-location').addEventListener('change', handleAutoLocation);
    document.getElementById('download-pdf-btn').addEventListener('click', downloadPDF);
    const locateBtn = document.getElementById('locate-btn');
    if (locateBtn) locateBtn.addEventListener('click', locateAndSearch);
}

function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.menu-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
    
    document.querySelectorAll('.mode-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${mode}-mode`).classList.add('active');
    
    if (mode === 'location') {
        clearResults();
    }
}

function handleAutoLocation() {
    const useAuto = document.getElementById('use-auto-location').checked;
    const locationInput = document.getElementById('location-input');
    locationInput.disabled = useAuto;
    
    if (useAuto) {
        locateAndSearch();
    }
}

function locateAndSearch() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const chk = document.getElementById('use-auto-location');
                if (chk && !chk.checked) {
                    chk.checked = true;
                    document.getElementById('location-input').disabled = true;
                }
                showStatus(`Detected Location: ${lat.toFixed(4)}, ${lon.toFixed(4)}`, 'success');
                searchHospitals(null, true, lat, lon);
            },
            error => {
                showStatus('Failed to get location: ' + error.message, 'error');
            }
        );
    } else {
        showStatus('Geolocation is not supported by your browser', 'error');
    }
}

async function handleLocationSearch() {
    const useAuto = document.getElementById('use-auto-location').checked;
    if (useAuto) {
        showStatus('Detecting your location...', 'success');
        locateAndSearch();
        return;
    }
    const locationInputVal = document.getElementById('location-input').value;
    if (!locationInputVal) {
        showStatus('Please enter a location', 'error');
        return;
    }
    showStatus('Searching for hospitals...', 'success');
    await searchHospitals(locationInputVal, false);
}

async function askSentBot() {
    const input = document.getElementById('sentbot-input');
    const ansDiv = document.getElementById('sentbot-answer');
    if (!input || !ansDiv) return;
    const q = input.value.trim();
    if (!q) {
        ansDiv.textContent = 'Please enter a question';
        ansDiv.className = 'status-message error';
        ansDiv.style.display = 'block';
        return;
    }
    const names = (window.currentHospitals || []).map(h => h.name);
    ansDiv.textContent = 'Thinking...';
    ansDiv.className = 'status-message success';
    ansDiv.style.display = 'block';
    try {
        const resp = await fetch('/api/sentbot-ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, hospital_names: names })
        });
        let data;
        try {
            data = await resp.json();
        } catch (_) {
            const text = await resp.text();
            throw new Error(text || 'Failed to get answer');
        }
        if (!resp.ok) throw new Error(data.detail || 'Failed to get answer');
        ansDiv.textContent = data.answer || 'No answer';
        ansDiv.className = 'status-message success';
        ansDiv.style.display = 'block';
    } catch (e) {
        ansDiv.textContent = 'Error: ' + e.message;
        ansDiv.className = 'status-message error';
        ansDiv.style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('sentbot-ask-btn');
    if (btn) btn.addEventListener('click', askSentBot);
});

async function searchHospitals(location, useAuto = false, lat = null, lon = null) {
    try {
        const response = await fetch('/api/analyze-location', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                location: location || '',
                use_auto: useAuto,
                lat: lat,
                lon: lon
            })
        });

        if (!response.ok) {
            try {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to fetch hospitals');
            } catch (_) {
                throw new Error('Failed to fetch hospitals');
            }
        }

        const data = await response.json();
        displayHospitals(data.hospitals, data.top_hospitals, lat || (data.hospitals[0]?.lat), lon || (data.hospitals[0]?.lon));
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

function displayHospitals(hospitals, topHospitals, centerLat, centerLon) {
    const hospitalsList = document.getElementById('hospitals-list');
    hospitalsList.innerHTML = '';

    if (hospitals.length === 0) {
        hospitalsList.innerHTML = '<div class="loading">No hospitals found</div>';
        return;
    }

    initMap(centerLat, centerLon, hospitals);

    hospitals.forEach(hospital => {
        const card = createHospitalCard(hospital);
        hospitalsList.appendChild(card);
    });

    if (topHospitals && topHospitals.length > 0) {
        const recommendedDiv = document.getElementById('recommended-hospitals');
        const recommendedList = document.getElementById('recommended-list');
        recommendedList.innerHTML = '';
        
        topHospitals.forEach(h => {
            const item = document.createElement('div');
            item.className = 'recommended-item';
            item.textContent = `${h.name} (Score: ${h.score.toFixed(2)})`;
            recommendedList.appendChild(item);
        });
        
        recommendedDiv.style.display = 'block';
    }

    window.currentHospitals = hospitals;
    const sentPanel = document.getElementById('sentbot-panel');
    if (sentPanel) sentPanel.style.display = 'block';
}

function createHospitalCard(hospital) {
    const card = document.createElement('div');
    card.className = 'hospital-card';
    
    const header = document.createElement('div');
    header.className = 'hospital-header';
    header.innerHTML = `
        <div class="hospital-name">🏥 ${hospital.name}</div>
        <div class="hospital-info">
            <span>⭐ ${hospital.rating}</span>
            <span>📝 ${hospital.total_reviews} reviews</span>
            <span>😊 ${(hospital.positive_ratio * 100).toFixed(2)}% positive</span>
            <span>🚗 ${hospital.distance_text}</span>
        </div>
    `;
    
    card.appendChild(header);
    
    if (hospital.aspect_summary && Object.keys(hospital.aspect_summary).length > 0) {
        const aspectDiv = document.createElement('div');
        aspectDiv.className = 'aspect-summary';
        aspectDiv.innerHTML = '<h4>📊 Sentiment Distribution by Aspect</h4>';
        const aspectList = document.createElement('div');
        Object.entries(hospital.aspect_summary).forEach(([aspect, counts]) => {
            const item = document.createElement('div');
            item.className = 'aspect-item';
            item.innerHTML = `
                <span><strong>${aspect.charAt(0).toUpperCase() + aspect.slice(1)}</strong></span>
                <span>Positive: ${counts.Positive} | Negative: ${counts.Negative}</span>
            `;
            aspectList.appendChild(item);
        });
        aspectDiv.appendChild(aspectList);
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        const canvas = document.createElement('canvas');
        const cid = `stacked-${Math.random().toString(36).slice(2)}`;
        canvas.id = cid;
        chartContainer.appendChild(canvas);
        aspectDiv.appendChild(chartContainer);
        card.appendChild(aspectDiv);
        const aspects = Object.keys(hospital.aspect_summary);
        const positiveData = aspects.map(a => hospital.aspect_summary[a].Positive || 0);
        const negativeData = aspects.map(a => hospital.aspect_summary[a].Negative || 0);
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: aspects.map(a => a.charAt(0).toUpperCase() + a.slice(1)),
                datasets: [
                    { label: 'Positive', data: positiveData, backgroundColor: 'rgba(40, 167, 69, 0.8)', stack: 'stack' },
                    { label: 'Negative', data: negativeData, backgroundColor: 'rgba(220, 53, 69, 0.8)', stack: 'stack' }
                ]
            },
            options: {
                responsive: true,
                scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } }
            }
        });
    }
    
    if (hospital.wordcloud) {
        const wcDiv = document.createElement('div');
        wcDiv.className = 'wordcloud-container';
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + hospital.wordcloud;
        wcDiv.appendChild(img);
        card.appendChild(wcDiv);
    }
    
    if (hospital.aspects && hospital.aspects.length > 0) {
        const reviewsDiv = document.createElement('div');
        reviewsDiv.className = 'sample-reviews';
        reviewsDiv.innerHTML = '<h4>📝 Sample Reviews by Aspect</h4>';
        
        const shown = new Set();
        hospital.aspects.forEach(([aspect, sentiment, line]) => {
            if (!shown.has(aspect)) {
                const aspectSection = document.createElement('div');
                aspectSection.innerHTML = `<h4>${aspect.charAt(0).toUpperCase() + aspect.slice(1)}</h4><ul></ul>`;
                const ul = aspectSection.querySelector('ul');
                
                hospital.aspects
                    .filter(([a, s, l]) => a === aspect)
                    .slice(0, 2)
                    .forEach(([a, s, l]) => {
                        const li = document.createElement('li');
                        li.textContent = `(${s}) ${l.trim()}`;
                        ul.appendChild(li);
                    });
                
                reviewsDiv.appendChild(aspectSection);
                shown.add(aspect);
            }
        });
        card.appendChild(reviewsDiv);
    }
    
    return card;
}

function initMap(lat, lon, hospitals) {
    const mapContainer = document.getElementById('map-container');
    mapContainer.style.display = 'block';
    if (!mapProvider) {
        mapProvider = (window.google && google.maps) ? 'google' : 'leaflet';
    }
    if (mapProvider === 'google') {
        if (!map) {
            map = new google.maps.Map(document.getElementById('map'), {
                center: { lat: lat, lng: lon },
                zoom: 12
            });
        } else {
            map.setCenter({ lat: lat, lng: lon });
        }
        markers.forEach(marker => marker.setMap(null));
        markers = [];
        hospitals.forEach(hospital => {
            const marker = new google.maps.Marker({
                position: { lat: hospital.lat, lng: hospital.lon },
                map: map,
                title: hospital.name
            });
            markers.push(marker);
        });
    } else {
        if (!map) {
            map = L.map('map').setView([lat, lon], 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '' }).addTo(map);
        } else {
            map.setView([lat, lon], 12);
        }
        markers.forEach(m => { try { map.removeLayer(m); } catch (e) {} });
        markers = [];
        hospitals.forEach(hospital => {
            const marker = L.marker([hospital.lat, hospital.lon]).addTo(map);
            marker.bindPopup(hospital.name);
            markers.push(marker);
        });
    }
}

async function handleCSVAnalysis() {
    const fileInput = document.getElementById('csv-file');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('Please select a CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showStatus('Analyzing reviews...', 'success');
        const response = await fetch('/api/analyze-csv', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            try {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to analyze CSV');
            } catch (_) {
                throw new Error('Failed to analyze CSV');
            }
        }

        const data = await response.json();
        displayCSVResults(data);
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

function displayCSVResults(data) {
    document.getElementById('csv-results').style.display = 'block';
    
    if (barChart) barChart.destroy();
    if (pieChart) pieChart.destroy();
    
    const aspectSummary = data.aspect_summary;
    const aspects = Object.keys(aspectSummary);
    const positiveData = aspects.map(a => aspectSummary[a].Positive);
    const negativeData = aspects.map(a => aspectSummary[a].Negative);
    
    const ctxBar = document.getElementById('bar-chart').getContext('2d');
    barChart = new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: aspects.map(a => a.charAt(0).toUpperCase() + a.slice(1)),
            datasets: [{
                label: 'Positive',
                data: positiveData,
                backgroundColor: 'rgba(40, 167, 69, 0.8)'
            }, {
                label: 'Negative',
                data: negativeData,
                backgroundColor: 'rgba(220, 53, 69, 0.8)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    const pieData = data.pie_chart;
    const ctxPie = document.getElementById('pie-chart').getContext('2d');
    pieChart = new Chart(ctxPie, {
        type: 'pie',
        data: {
            labels: pieData.map(d => d.aspect.charAt(0).toUpperCase() + d.aspect.slice(1)),
            datasets: [{
                data: pieData.map(d => d.count),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(199, 199, 199, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true
        }
    });
    
    if (data.wordcloud) {
        document.getElementById('wordcloud-img').src = 'data:image/png;base64,' + data.wordcloud;
    }
    
    window.csvAnalysisData = data;
}

async function downloadPDF() {
    if (!window.csvAnalysisData) {
        showStatus('Please analyze a CSV file first', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                aspect_summary: window.csvAnalysisData.aspect_summary,
                aspects: window.csvAnalysisData.aspects
            })
        });

        if (!response.ok) {
            throw new Error('Failed to generate PDF');
        }

        const data = await response.json();
        const pdfBlob = base64ToBlob(data.pdf, 'application/pdf');
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'hospital_feedback.pdf';
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('location-status');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;
    statusDiv.style.display = 'block';
}

function clearResults() {
    document.getElementById('hospitals-list').innerHTML = '';
    document.getElementById('recommended-hospitals').style.display = 'none';
    document.getElementById('map-container').style.display = 'none';
}

