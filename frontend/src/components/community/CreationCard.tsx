import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Heart, 
  MessageCircle, 
  Share2, 
  Eye, 
  TrendingUp, 
  Sparkles,
  User,
  Calendar,
  Hash,
  BarChart3,
  Target
} from 'lucide-react';
import { Card, CardContent, CardHeader } from '../ui/card';
import { CommunityCreation } from '../../types';

interface CreationCardProps {
  creation: CommunityCreation;
  currentUserId: string;
  onLike: () => void;
  onShare: (platform: string) => void;
  onComment?: () => void;
  onView?: () => void;
  className?: string;
  showFullContent?: boolean;
}

export const CreationCard: React.FC<CreationCardProps> = ({
  creation,
  currentUserId,
  onLike,
  onShare,
  onComment,
  onView,
  className = '',
  showFullContent = false
}) => {
  const [isExpanded, setIsExpanded] = useState(showFullContent);
  const [shareMenuOpen, setShareMenuOpen] = useState(false);

  const {
    id,
    title,
    contentPreview,
    fullContent,
    creatorUsername,
    creatorDisplayName,
    creatorPrimaryMuse,
    formType,
    theme,
    mathematicalFitness,
    semanticCoherence,
    archetypalResonance,
    discoveryCoordinates,
    likes,
    comments,
    shares,
    views,
    isLiked,
    createdAt,
    tags
  } = creation;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    if (diffInHours < 168) return `${Math.floor(diffInHours / 24)}d ago`;
    return date.toLocaleDateString();
  };

  const getMuseColor = (muse: string) => {
    const colors: Record<string, string> = {
      'CALLIOPE': 'text-red-500',
      'ERATO': 'text-pink-500',
      'URANIA': 'text-blue-500',
      'THALIA': 'text-yellow-500',
      'MELPOMENE': 'text-purple-500',
      'POLYHYMNIA': 'text-indigo-500',
      'TERPSICHORE': 'text-green-500',
      'EUTERPE': 'text-teal-500',
      'CLIO': 'text-orange-500',
      'SOPHIA': 'text-violet-500',
      'TECHNE': 'text-cyan-500',
      'PSYCHE': 'text-rose-500'
    };
    return colors[muse] || 'text-gray-500';
  };

  const getFormTypeIcon = (form: string) => {
    const icons: Record<string, React.ReactNode> = {
      'sonnet': <Hash className="h-4 w-4" />,
      'haiku': <Sparkles className="h-4 w-4" />,
      'villanelle': <TrendingUp className="h-4 w-4" />,
      'free_verse': <Eye className="h-4 w-4" />
    };
    return icons[form] || <Hash className="h-4 w-4" />;
  };

  const getFitnessColor = (fitness: number) => {
    if (fitness >= 0.9) return 'text-emerald-500';
    if (fitness >= 0.8) return 'text-green-500';
    if (fitness >= 0.7) return 'text-yellow-500';
    if (fitness >= 0.6) return 'text-orange-500';
    return 'text-red-500';
  };

  const shareOptions = [
    { platform: 'twitter', label: 'Twitter', icon: '🐦' },
    { platform: 'facebook', label: 'Facebook', icon: '📘' },
    { platform: 'copy', label: 'Copy Link', icon: '🔗' },
    { platform: 'email', label: 'Email', icon: '✉️' }
  ];

  const handleShare = (platform: string) => {
    onShare(platform);
    setShareMenuOpen(false);
  };

  const contentToDisplay = isExpanded ? fullContent : contentPreview;
  const needsExpansion = fullContent && fullContent.length > contentPreview.length;

  return (
    <Card className={`bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-purple-200 dark:border-purple-800 hover:shadow-xl transition-all duration-300 ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 text-white font-bold text-sm">
              {creatorDisplayName?.[0] || creatorUsername[0]}
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h4 className="font-semibold text-gray-900 dark:text-white">
                  {creatorDisplayName || creatorUsername}
                </h4>
                <span className={`text-sm font-medium ${getMuseColor(creatorPrimaryMuse)}`}>
                  {creatorPrimaryMuse}
                </span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                <Calendar className="h-3 w-3" />
                <span>{formatDate(createdAt)}</span>
                <span>•</span>
                <div className="flex items-center space-x-1">
                  {getFormTypeIcon(formType)}
                  <span className="capitalize">{formType.replace('_', ' ')}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <div className="text-right">
              <div className={`text-sm font-bold ${getFitnessColor(mathematicalFitness)}`}>
                {(mathematicalFitness * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500">fitness</div>
            </div>
            <BarChart3 className={`h-4 w-4 ${getFitnessColor(mathematicalFitness)}`} />
          </div>
        </div>

        {title && (
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mt-2">
            {title}
          </h3>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        {/* Content */}
        <div className="mb-4">
          <div className="relative">
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <pre className="whitespace-pre-wrap font-serif text-gray-800 dark:text-gray-200 leading-relaxed">
                {contentToDisplay}
              </pre>
            </div>
            
            {needsExpansion && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="mt-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 text-sm font-medium transition-colors"
              >
                {isExpanded ? 'Show less' : 'Read more'}
              </button>
            )}
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-3 gap-4 mb-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div className="text-center">
            <div className="text-sm font-bold text-gray-900 dark:text-white">
              {(mathematicalFitness * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Mathematical
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-gray-900 dark:text-white">
              {(semanticCoherence * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Semantic
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-gray-900 dark:text-white">
              {(archetypalResonance * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Resonance
            </div>
          </div>
        </div>

        {/* Discovery Coordinates */}
        {discoveryCoordinates && (
          <div className="mb-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="h-4 w-4 text-purple-600 dark:text-purple-400" />
              <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
                Discovery Coordinates
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {discoveryCoordinates.sacredConstant && (
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Constant:</span>
                  <span className="ml-1 font-medium text-purple-600 dark:text-purple-400">
                    {discoveryCoordinates.sacredConstant}
                  </span>
                </div>
              )}
              {discoveryCoordinates.theme && (
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Theme:</span>
                  <span className="ml-1 font-medium text-purple-600 dark:text-purple-400 capitalize">
                    {discoveryCoordinates.theme}
                  </span>
                </div>
              )}
              {discoveryCoordinates.entropySeed && (
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Entropy:</span>
                  <span className="ml-1 font-medium text-purple-600 dark:text-purple-400">
                    {discoveryCoordinates.entropySeed.toFixed(3)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tags */}
        {tags && tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {tags.map((tag, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded-full border border-blue-200 dark:border-blue-800"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLike}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg transition-colors ${
                isLiked
                  ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 dark:hover:text-red-400'
              }`}
            >
              <Heart className={`h-4 w-4 ${isLiked ? 'fill-current' : ''}`} />
              <span className="text-sm font-medium">{likes}</span>
            </motion.button>

            {onComment && (
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onComment}
                className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <MessageCircle className="h-4 w-4" />
                <span className="text-sm font-medium">{comments}</span>
              </motion.button>
            )}

            <div className="relative">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShareMenuOpen(!shareMenuOpen)}
                className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-green-50 dark:hover:bg-green-900/20 hover:text-green-600 dark:hover:text-green-400 transition-colors"
              >
                <Share2 className="h-4 w-4" />
                <span className="text-sm font-medium">{shares}</span>
              </motion.button>

              {shareMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute bottom-full mb-2 left-0 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-2 min-w-[150px] z-10"
                >
                  {shareOptions.map((option) => (
                    <button
                      key={option.platform}
                      onClick={() => handleShare(option.platform)}
                      className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center space-x-2 text-sm text-gray-700 dark:text-gray-300"
                    >
                      <span>{option.icon}</span>
                      <span>{option.label}</span>
                    </button>
                  ))}
                </motion.div>
              )}
            </div>
          </div>

          <div className="flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
            <Eye className="h-3 w-3" />
            <span>{views.toLocaleString()} views</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};