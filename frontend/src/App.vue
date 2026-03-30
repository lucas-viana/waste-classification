<script setup>
import { computed, nextTick, onBeforeUnmount, ref } from "vue";

const selectedFile = ref(null);
const previewUrl = ref("");
const isLoading = ref(false);
const errorMessage = ref("");
const result = ref(null);

const cameraOn = ref(false);
const isLivePredicting = ref(false);
const cameraStream = ref(null);
const videoRef = ref(null);

const liveIntervalMs = 1500;
let liveTimerId = null;
let captureCanvas = null;

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const probabilities = computed(() => {
  if (!result.value || !result.value.probabilities) return [];
  return [...result.value.probabilities].sort((a, b) => b.probability - a.probability);
});

function clearPreviewUrl() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = "";
  }
}

async function classifyBlob(fileOrBlob, fallbackName = "imagem.jpg") {
  isLoading.value = true;
  errorMessage.value = "";

  try {
    const formData = new FormData();
    const uploadFile = fileOrBlob instanceof File
      ? fileOrBlob
      : new File([fileOrBlob], fallbackName, { type: fileOrBlob.type || "image/jpeg" });
    formData.append("file", uploadFile);

    const response = await fetch(`${apiBaseUrl}/api/predict`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Falha ao classificar a imagem.");
    }

    result.value = await response.json();
  } catch (error) {
    errorMessage.value = error.message || "Erro inesperado ao classificar imagem.";
  } finally {
    isLoading.value = false;
  }
}

function stopLivePrediction() {
  isLivePredicting.value = false;
  if (liveTimerId) {
    clearInterval(liveTimerId);
    liveTimerId = null;
  }
}

function closeCamera() {
  stopLivePrediction();

  if (cameraStream.value) {
    for (const track of cameraStream.value.getTracks()) {
      track.stop();
    }
  }

  cameraStream.value = null;
  cameraOn.value = false;

  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }
}

async function openCamera() {
  if (cameraOn.value) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    errorMessage.value = "Seu navegador nao suporta acesso a camera.";
    return;
  }

  try {
    errorMessage.value = "";
    result.value = null;
    selectedFile.value = null;
    clearPreviewUrl();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });

    cameraStream.value = stream;
    cameraOn.value = true;

    await nextTick();
    if (videoRef.value) {
      videoRef.value.srcObject = stream;
      await videoRef.value.play();
    }
  } catch {
    errorMessage.value = "Nao foi possivel abrir a camera. Verifique as permissoes.";
    closeCamera();
  }
}

function onFileChange(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  closeCamera();

  selectedFile.value = file;
  errorMessage.value = "";
  result.value = null;

  clearPreviewUrl();
  previewUrl.value = URL.createObjectURL(file);
}

async function classifyImage() {
  if (!selectedFile.value) {
    errorMessage.value = "Selecione uma imagem antes de classificar.";
    return;
  }

  await classifyBlob(selectedFile.value, selectedFile.value.name || "upload.jpg");
}

async function classifyCameraFrame() {
  if (!cameraOn.value || !videoRef.value) {
    errorMessage.value = "Abra a camera antes de classificar em tempo real.";
    return;
  }

  if (videoRef.value.videoWidth === 0 || videoRef.value.videoHeight === 0) {
    return;
  }

  if (!captureCanvas) {
    captureCanvas = document.createElement("canvas");
  }

  captureCanvas.width = videoRef.value.videoWidth;
  captureCanvas.height = videoRef.value.videoHeight;

  const context = captureCanvas.getContext("2d");
  if (!context) {
    errorMessage.value = "Nao foi possivel processar o frame da camera.";
    return;
  }

  context.drawImage(videoRef.value, 0, 0, captureCanvas.width, captureCanvas.height);

  const blob = await new Promise((resolve) => {
    captureCanvas.toBlob(resolve, "image/jpeg", 0.92);
  });

  if (!blob) {
    errorMessage.value = "Nao foi possivel capturar imagem da camera.";
    return;
  }

  await classifyBlob(blob, "captura_camera.jpg");
}

async function startLivePrediction() {
  if (!cameraOn.value) {
    await openCamera();
    if (!cameraOn.value) return;
  }

  if (isLivePredicting.value) return;

  isLivePredicting.value = true;
  await classifyCameraFrame();

  liveTimerId = setInterval(() => {
    if (!isLoading.value) {
      classifyCameraFrame();
    }
  }, liveIntervalMs);
}

onBeforeUnmount(() => {
  stopLivePrediction();
  closeCamera();
  clearPreviewUrl();
});
</script>

<template>
  <main class="page-shell">
    <div class="aura aura-left" aria-hidden="true"></div>
    <div class="aura aura-right" aria-hidden="true"></div>

    <section class="panel">
      <header class="hero">
        <p class="eyebrow">TensorFlow + Keras + FastAPI</p>
        <h1>Classificação de Resíduos</h1>
        <p class="subtitle">
          Envie uma imagem ou use a camera para classificar residuos com seu modelo treinado.
        </p>
      </header>

      <div class="uploader">
        <label for="file-input" class="file-trigger">Escolher Imagem</label>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          @change="onFileChange"
        />

        <button class="predict-button" :disabled="isLoading" @click="classifyImage">
          <span v-if="isLoading">Classificando...</span>
          <span v-else>Classificar Imagem</span>
        </button>

        <button v-if="!cameraOn" class="camera-button" :disabled="isLoading" @click="openCamera">
          Abrir Camera
        </button>
        <button v-else class="camera-button close" :disabled="isLoading" @click="closeCamera">
          Fechar Camera
        </button>

        <button v-if="cameraOn" class="camera-button" :disabled="isLoading" @click="classifyCameraFrame">
          Classificar Camera
        </button>

        <button
          v-if="cameraOn && !isLivePredicting"
          class="camera-button live"
          :disabled="isLoading"
          @click="startLivePrediction"
        >
          Iniciar Tempo Real
        </button>
        <button
          v-if="cameraOn && isLivePredicting"
          class="camera-button live stop"
          :disabled="isLoading"
          @click="stopLivePrediction"
        >
          Parar Tempo Real
        </button>
      </div>

      <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>

      <div class="content-grid">
        <article class="card preview-card">
          <h2>Pre-visualizacao</h2>
          <div class="preview-frame">
            <template v-if="cameraOn">
              <video ref="videoRef" autoplay muted playsinline></video>
              <p class="camera-hint">Aponte o residuo para a camera.</p>
              <span v-if="isLivePredicting" class="camera-live-tag">Tempo real ativo</span>
            </template>
            <img
              v-else-if="previewUrl"
              :src="previewUrl"
              alt="Residuo selecionado"
            />
            <p v-else>Nenhuma imagem selecionada.</p>
          </div>
        </article>

        <article class="card result-card">
          <h2>Resultado da Classificacao</h2>

          <div v-if="result" class="result-body">
            <p class="winner-label">Classe principal</p>
            <p class="winner-value">{{ result.predicted_class }}</p>
            <p class="winner-confidence">
              Confianca: {{ (result.confidence * 100).toFixed(2) }}%
            </p>

            <ul class="probability-list">
              <li v-for="item in probabilities" :key="item.class_name">
                <div class="probability-head">
                  <span>{{ item.class_name }}</span>
                  <span>{{ (item.probability * 100).toFixed(2) }}%</span>
                </div>
                <div class="probability-track">
                  <div
                    class="probability-fill"
                    :style="{ width: `${item.probability * 100}%` }"
                  ></div>
                </div>
              </li>
            </ul>
          </div>

          <p v-else class="placeholder-text">
            Execute uma classificacao para ver as probabilidades por classe.
          </p>
        </article>
      </div>
    </section>
  </main>
</template>
