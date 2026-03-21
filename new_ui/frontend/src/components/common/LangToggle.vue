<template>
  <el-tooltip :content="appStore.locale === 'zh' ? 'Switch to English' : '切换到中文'">
    <el-button circle size="small" @click="toggle">
      <span style="font-size: 13px; font-weight: 600;">{{ appStore.locale === 'zh' ? 'En' : '中' }}</span>
    </el-button>
  </el-tooltip>
</template>

<script setup lang="ts">
import { useAppStore } from '@/stores/app'
import { useI18n } from 'vue-i18n'

const appStore = useAppStore()
const { locale } = useI18n()

function toggle() {
  appStore.toggleLocale()
  locale.value = appStore.locale
}
</script>
