<template>
  <div class="behavior-wrap">
    <el-input
      v-model="question"
      type="textarea"
      :rows="2"
      :placeholder="$t('results.behaviorQuestionPh')"
      style="margin-bottom: 12px;"
    />
    <el-button type="primary" size="small" :disabled="labels.length < 2" :loading="loading" @click="load">
      {{ $t('results.behaviorLoad') }}
    </el-button>
    <p v-if="note" class="note">{{ note }}</p>

    <!-- 指标 delta 区域 -->
    <div v-if="metricDeltas.length" class="delta-section">
      <div class="sub">{{ $t('results.metricDelta') }}</div>
      <div class="delta-grid">
        <div v-for="d in metricDeltas" :key="d.key" class="delta-item">
          <span class="delta-name">{{ d.name }}</span>
          <span class="delta-val before">{{ d.before }}</span>
          <span class="delta-arrow" :class="d.direction">
            {{ d.direction === 'up' ? '↑' : d.direction === 'down' ? '↓' : '—' }}
          </span>
          <span class="delta-val after">{{ d.after }}</span>
          <span class="delta-diff" :class="d.direction">({{ d.direction === 'up' ? '+' : '' }}{{ d.delta }})</span>
        </div>
      </div>
    </div>

    <!-- 多样本对比区域 -->
    <div v-if="samples.length" class="samples-section">
      <div class="sample-nav">
        <el-button size="small" :disabled="sampleIdx <= 0" @click="sampleIdx--">{{ $t('results.prevSample') }}</el-button>
        <span class="sample-counter">{{ $t('results.sampleOf', { n: sampleIdx + 1, t: samples.length }) }}</span>
        <el-button size="small" :disabled="sampleIdx >= samples.length - 1" @click="sampleIdx++">{{ $t('results.nextSample') }}</el-button>
      </div>
      <div class="sample-question">
        <strong>Q:</strong> {{ currentSample?.question || '—' }}
      </div>
      <div class="cols">
        <div class="col">
          <div class="col-title">{{ $t('results.before') }} — {{ payload?.before?.label }}</div>
          <pre class="block" v-html="diffHtml(currentSample?.before_answer || '', currentSample?.after_answer || '', 'before')"></pre>
        </div>
        <div class="col">
          <div class="col-title">{{ $t('results.after') }} — {{ payload?.after?.label }}</div>
          <pre class="block" v-html="diffHtml(currentSample?.before_answer || '', currentSample?.after_answer || '', 'after')"></pre>
        </div>
      </div>
    </div>

    <!-- 兜底：无样本时显示指标摘要 -->
    <div v-else-if="payload" class="cols">
      <div class="col">
        <div class="col-title">{{ $t('results.before') }} — {{ payload.before?.label }}</div>
        <pre class="block">{{ payload.before?.metrics_text || '—' }}</pre>
      </div>
      <div class="col">
        <div class="col-title">{{ $t('results.after') }} — {{ payload.after?.label }}</div>
        <pre class="block">{{ payload.after?.metrics_text || '—' }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { resultsApi } from '@/api'

interface Sample {
  question: string
  before_answer: string
  after_answer: string
}

interface MetricDelta {
  name: string
  key: string
  before: string
  after: string
  delta: string
  direction: 'up' | 'down' | 'same'
}

const props = defineProps<{ labels: string[] }>()

const question = ref('')
const loading = ref(false)
const payload = ref<any>(null)
const note = ref('')
const samples = ref<Sample[]>([])
const metricDeltas = ref<MetricDelta[]>([])
const sampleIdx = ref(0)

const currentSample = computed(() => samples.value[sampleIdx.value] || null)

watch(
  () => props.labels.join('|'),
  () => {
    payload.value = null
    note.value = ''
    samples.value = []
    metricDeltas.value = []
    sampleIdx.value = 0
  },
)

async function load() {
  if (props.labels.length < 2) return
  loading.value = true
  note.value = ''
  try {
    const data = await resultsApi.behaviorCompare(props.labels.slice(0, 2), question.value)
    payload.value = data
    note.value = data.note || ''
    samples.value = data.samples || []
    metricDeltas.value = data.metric_deltas || []
    sampleIdx.value = 0
    if (data.question && !question.value) question.value = data.question
  } catch {
    note.value = 'Failed to load'
  } finally {
    loading.value = false
  }
}

/**
 * 简单的字符级 diff 高亮：将两段文本中不同的部分用 <mark> 包裹。
 * side='before' 高亮 before 中独有的部分，side='after' 高亮 after 中独有的部分。
 */
function diffHtml(before: string, after: string, side: 'before' | 'after'): string {
  const text = side === 'before' ? before : after
  const other = side === 'before' ? after : before
  if (!text) return '—'
  if (text === other || !other) return escHtml(text)

  // 按词分割做简单 diff
  const tWords = text.split(/(\s+)/)
  const oWords = other.split(/(\s+)/)
  const oSet = new Set(oWords)
  const parts: string[] = []
  for (const w of tWords) {
    if (oSet.has(w)) {
      parts.push(escHtml(w))
    } else {
      parts.push(`<mark>${escHtml(w)}</mark>`)
    }
  }
  return parts.join('')
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
</script>

<style scoped>
.behavior-wrap { padding: 8px 0; }
.note {
  font-size: 12px;
  color: var(--text-muted);
  margin: 8px 0;
}
.delta-section {
  margin-top: 16px;
  padding: 10px;
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
}
.delta-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  margin-top: 6px;
}
.delta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
}
.delta-name {
  color: var(--text-secondary);
  min-width: 100px;
}
.delta-val { font-family: monospace; }
.delta-val.before { color: var(--text-muted); }
.delta-val.after { font-weight: 600; }
.delta-arrow { font-weight: 700; }
.delta-arrow.up { color: #52c41a; }
.delta-arrow.down { color: #ff4d4f; }
.delta-arrow.same { color: var(--text-muted); }
.delta-diff { font-size: 11px; }
.delta-diff.up { color: #52c41a; }
.delta-diff.down { color: #ff4d4f; }
.delta-diff.same { color: var(--text-muted); }
.samples-section { margin-top: 16px; }
.sample-nav {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.sample-counter {
  font-size: 12px;
  color: var(--text-secondary);
}
.sample-question {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  padding: 6px 8px;
  background: var(--bg-surface);
  border-radius: var(--radius-sm);
}
.cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.col-title {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 8px;
}
.sub {
  font-size: 12px;
  color: var(--text-muted);
  margin: 0 0 4px;
  font-weight: 600;
}
.block {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 10px;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  max-height: 220px;
  overflow: auto;
}
.block :deep(mark) {
  background: #fff3b0;
  border-radius: 2px;
  padding: 0 1px;
}
@media (max-width: 900px) {
  .cols { grid-template-columns: 1fr; }
}
</style>
