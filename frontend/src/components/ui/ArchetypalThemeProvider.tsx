/**
 * Archetypal Theme Provider
 * 
 * Provides dynamic theming based on the user's archetypal signature,
 * applying sacred geometry patterns and muse-specific color schemes.
 */

import React, { useEffect } from 'react'
import { ThemeConfig } from '@/types'

interface ArchetypalThemeProviderProps {
  theme: ThemeConfig
  children: React.ReactNode
}

const ArchetypalThemeProvider: React.FC<ArchetypalThemeProviderProps> = ({
  theme,
  children,
}) => {
  useEffect(() => {
    // Apply theme to CSS custom properties
    const root = document.documentElement
    const { color_palette } = theme
    
    // Convert hex colors to HSL for CSS variables
    const hexToHsl = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
      if (!result) return '0 0% 0%'
      
      const r = parseInt(result[1], 16) / 255
      const g = parseInt(result[2], 16) / 255
      const b = parseInt(result[3], 16) / 255
      
      const max = Math.max(r, g, b)
      const min = Math.min(r, g, b)
      let h = 0
      let s = 0
      const l = (max + min) / 2
      
      if (max !== min) {
        const d = max - min
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
        
        switch (max) {
          case r: h = (g - b) / d + (g < b ? 6 : 0); break
          case g: h = (b - r) / d + 2; break
          case b: h = (r - g) / d + 4; break
        }
        h /= 6
      }
      
      return `${Math.round(h * 360)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`
    }
    
    // Apply primary color
    root.style.setProperty('--primary', hexToHsl(color_palette.primary))
    root.style.setProperty('--primary-foreground', hexToHsl(color_palette.background))
    
    // Apply secondary color
    root.style.setProperty('--secondary', hexToHsl(color_palette.secondary))
    root.style.setProperty('--secondary-foreground', hexToHsl(color_palette.text))
    
    // Apply accent color
    root.style.setProperty('--accent', hexToHsl(color_palette.accent))
    root.style.setProperty('--accent-foreground', hexToHsl(color_palette.text))
    
    // Apply archetypal class to body
    document.body.className = document.body.className.replace(
      /theme-\w+/g,
      ''
    )
    document.body.classList.add(`theme-${theme.primary_muse.toLowerCase()}`)
    
    // Apply sacred geometry pattern
    root.style.setProperty('--sacred-pattern', theme.sacred_geometry_pattern)
    
    // Apply font family
    const fontClass = {
      'serif': 'font-serif',
      'sans': 'font-sans',
      'mono': 'font-mono',
      'display': 'font-display',
    }[theme.font_family] || 'font-sans'
    
    document.body.className = document.body.className.replace(
      /font-\w+/g,
      ''
    )
    document.body.classList.add(fontClass)
    
    // Apply animation style
    const animationClass = {
      'subtle': 'animations-subtle',
      'moderate': 'animations-moderate',
      'dynamic': 'animations-dynamic',
    }[theme.animation_style] || 'animations-moderate'
    
    document.body.className = document.body.className.replace(
      /animations-\w+/g,
      ''
    )
    document.body.classList.add(animationClass)
    
  }, [theme])
  
  return (
    <div className="archetypal-theme-provider">
      {children}
    </div>
  )
}

export default ArchetypalThemeProvider