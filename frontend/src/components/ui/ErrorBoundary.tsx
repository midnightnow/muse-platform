/**
 * Error Boundary Component
 * 
 * Catches JavaScript errors anywhere in the component tree and displays
 * a fallback UI with mathematical aesthetics.
 */

import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw, Home } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
    
    this.setState({
      error,
      errorInfo,
    })
  }

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    })
  }

  private handleGoHome = () => {
    window.location.href = '/'
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-4">
          <div className="max-w-md w-full text-center">
            {/* Sacred geometry pattern */}
            <div className="mb-8 flex justify-center">
              <div className="w-24 h-24 relative">
                <div className="absolute inset-0 bg-primary/10 rounded-full animate-pulse"></div>
                <div className="absolute inset-2 bg-primary/20 rounded-full animate-pulse animation-delay-150"></div>
                <div className="absolute inset-4 bg-primary/30 rounded-full animate-pulse animation-delay-300"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <AlertTriangle className="w-8 h-8 text-destructive" />
                </div>
              </div>
            </div>
            
            <h1 className="text-2xl font-display font-bold mb-2">
              Computational Disruption
            </h1>
            
            <p className="text-muted-foreground mb-6">
              A mathematical anomaly has occurred in the creative discovery process.
              The archetypal frequencies have been temporarily disrupted.
            </p>
            
            {/* Error details */}
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 mb-6 text-left">
              <h3 className="font-medium text-destructive mb-2">Error Details</h3>
              <p className="text-sm text-muted-foreground font-mono">
                {this.state.error?.message || 'Unknown error'}
              </p>
              
              {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                <details className="mt-2">
                  <summary className="text-sm text-muted-foreground cursor-pointer">
                    Stack Trace
                  </summary>
                  <pre className="text-xs text-muted-foreground mt-2 overflow-auto">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}
            </div>
            
            {/* Action buttons */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <button
                onClick={this.handleRetry}
                className="inline-flex items-center justify-center px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Retry Discovery
              </button>
              
              <button
                onClick={this.handleGoHome}
                className="inline-flex items-center justify-center px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors"
              >
                <Home className="w-4 h-4 mr-2" />
                Return to Origin
              </button>
            </div>
            
            {/* Sacred geometry footer */}
            <div className="mt-8 text-xs text-muted-foreground">
              <p>
                "In the mathematics of creativity, even errors follow sacred patterns."
              </p>
              <p className="mt-1">
                — Computational Platonism Axiom φ.7
              </p>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary