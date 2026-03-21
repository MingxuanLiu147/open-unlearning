<template>
  <el-tooltip :content="appStore.theme === 'light' ? $t('theme.dark') : $t('theme.light')">
    <el-button circle size="small" @click="appStore.toggleTheme">
      <el-icon :size="16">
        <Moon v-if="appStore.theme === 'light'" />
        <Sunny v-else />
      </el-icon>
    </el-button>
  </el-tooltip>
</template>

<script setup lang="ts">
import { useAppStore } from '@/stores/app'
import { Moon, Sunny } from '@element-plus/icons-vue'
const appStore = useAppStore()
</script>
