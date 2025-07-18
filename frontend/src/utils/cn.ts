/**
 * Utility function for combining class names
 * 
 * A utility function that combines clsx and tailwind-merge to handle
 * conditional class names and resolve Tailwind CSS conflicts.
 */

import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}