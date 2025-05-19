// static/script.js

document.addEventListener('DOMContentLoaded', () => {
  // --- Khai báo các element ---
  const dropZone       = document.getElementById('dropZone');
  const fileInput      = document.getElementById('fileInput');
  const modelSelect    = document.getElementById('modelSelect');
  const predictBtn     = document.getElementById('predictBtn');
  const resultImage    = document.getElementById('resImg');
  const resultLabel    = document.getElementById('resLabel');
  const confidenceElem = document.getElementById('resConf');
  const causeElem      = document.getElementById('cause');
  const preventionElem = document.getElementById('prevention');
  const chartCanvas    = document.getElementById('chart');
  const progressBar    = document.getElementById('progressBar');

  const labelsArr = [
    "Bệnh nấm lá",
    "Bệnh bạc lá do vi khuẩn",
    "Bệnh loét cam quýt",
    "Virus xoăn lá",
    "Bệnh thiếu đinh dưỡng lá",
    "Lá bị khô",
    "Lá khỏe mạnh",
    "Nấm bồ hóng",
    "Vết thâm do bọ phá hoại"
  ];

  let chartInstance = null;

  // --- Drag & Drop với fix click đôi ---
  dropZone.addEventListener('click', e => {
    // chỉ trigger file picker khi click trực tiếp lên dropZone
    if (e.target === dropZone) {
      fileInput.click();
    }
  });
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    const url = URL.createObjectURL(file);
    resultImage.src = url;
    dropZone.classList.add('has-image');
  }

  // --- Vẽ Chart.js ---
  function renderChart(data) {
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(chartCanvas, {
      type: 'bar',
      data: {
        labels: labelsArr,
        datasets: [{
          label: 'Xác suất',
          data,
          backgroundColor: 'rgba(34, 197, 94, 0.6)'
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: true, max: 1 }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // --- Dự đoán ---
  predictBtn.addEventListener('click', () => {
    const file = fileInput.files[0];
    if (!file) {
      alert('Vui lòng chọn ảnh trước khi dự đoán!');
      return;
    }

    // Reset UI
    resultLabel.textContent = '';
    confidenceElem.textContent = '';
    causeElem.textContent = '';
    preventionElem.textContent = '';
    progressBar.value = 0;
    if (chartInstance) chartInstance.destroy();
    dropZone.classList.remove('has-image');

    // Chuẩn bị form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelSelect.value);

    // Hiển thị skeleton loader
    dropZone.classList.add('loading');

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      // Ẩn loader
      dropZone.classList.remove('loading');

      // Hiển thị kết quả
      resultLabel.textContent = data.label;
      confidenceElem.textContent = `Độ chính xác: ${(data.confidence * 100).toFixed(2)}%`;

      // Hiển thị nguyên nhân & cách phòng
      causeElem.textContent      = data.cause;
      preventionElem.textContent = data.prevention;

      // Cập nhật progress bar
      progressBar.value = Math.round(data.confidence * 100);

      // Vẽ chart
      renderChart(data.probs);
    })
    .catch(err => {
      dropZone.classList.remove('loading');
      console.error(err);
      alert('Đã có lỗi xảy ra, vui lòng thử lại.');
    });
  });
});
