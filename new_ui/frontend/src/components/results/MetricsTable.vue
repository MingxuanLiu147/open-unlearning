<template>
  <div>
    <div v-if="Object.keys(data).length === 0" class="empty">{{ $t('results.noData') }}</div>
    <div v-for="(metrics, evalName) in data" :key="evalName" class="eval-block">
      <h4 class="eval-name">{{ evalName }}</h4>
      <el-table :data="tableRows(metrics as Record<string, Record<string, number>>)" stripe size="small" border>
        <el-table-column prop="metric" :label="$t('results.metrics')" width="200" fixed />
        <el-table-column
          v-for="lbl in labels"
          :key="lbl"
          :label="shortLabel(lbl)"
          align="center"
          min-width="120"
        >
          <template #default="{ row }">
            <span :class="{ best: isBest(row, lbl) }">
              {{ formatVal(row[lbl]) }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  data: Record<string, any>
  labels: string[]
}>()

function shortLabel(l: string) {
  return l.split('/').pop() || l
}

function tableRows(metrics: Record<string, Record<string, number>>) {
  return Object.entries(metrics).map(([metric, values]) => ({ metric, ...values }))
}

function formatVal(v: any) {
  if (v == null) return '—'
  if (typeof v === 'number') return Math.abs(v) < 0.01 || Math.abs(v) > 1000 ? v.toExponential(3) : v.toFixed(4)
  return String(v)
}

function isBest(row: any, label: string) {
  const val = row[label]
  if (typeof val !== 'number') return false
  const nums = props.labels.map(l => row[l]).filter(v => typeof v === 'number')
  return val === Math.max(...nums)
}
</script>

<style scoped>
.eval-block { margin-bottom: 20px; }
.eval-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--accent-primary);
  border-left: 3px solid var(--accent-primary);
  padding-left: 8px;
  margin-bottom: 8px;
}
.best { font-weight: 700; color: var(--accent-success); }
.empty { color: var(--text-muted); text-align: center; padding: 40px; }
</style>
