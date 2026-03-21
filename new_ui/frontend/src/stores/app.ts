import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useAppStore = defineStore('app', () => {
  const theme = ref<'light' | 'dark'>(
    (localStorage.getItem('ks-theme') as 'light' | 'dark') || 'light'
  )
  const locale = ref<'zh' | 'en'>(
    (localStorage.getItem('ks-locale') as 'zh' | 'en') || 'zh'
  )
  const copilotOpen = ref(false)

  function toggleTheme() {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }

  function toggleLocale() {
    locale.value = locale.value === 'zh' ? 'en' : 'zh'
  }

  function toggleCopilot() {
    copilotOpen.value = !copilotOpen.value
  }

  watch(theme, (v) => {
    document.documentElement.setAttribute('data-theme', v)
    localStorage.setItem('ks-theme', v)
  }, { immediate: true })

  watch(locale, (v) => {
    localStorage.setItem('ks-locale', v)
  })

  return { theme, locale, copilotOpen, toggleTheme, toggleLocale, toggleCopilot }
})
