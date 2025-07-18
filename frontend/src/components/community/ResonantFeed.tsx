import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Heart, MessageCircle, Share2, Filter, Search, Sparkles, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { CreationCard } from './CreationCard';
import { useMuseAPI } from '../../hooks/useMuseAPI';
import { CommunityCreation, UserProfile, FilterOptions } from '../../types';

interface ResonantFeedProps {
  userId: string;
  className?: string;
}

interface FeedStats {
  totalCreations: number;
  averageResonance: number;
  topMuses: Array<{ muse: string; count: number }>;
  trendingThemes: Array<{ theme: string; growth: number }>;
}

export const ResonantFeed: React.FC<ResonantFeedProps> = ({ userId, className = '' }) => {
  const [creations, setCreations] = useState<CommunityCreation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterOptions>({
    formType: 'all',
    minFitness: 0,
    theme: 'all',
    sortBy: 'resonance',
    timeRange: 'week'
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [feedStats, setFeedStats] = useState<FeedStats | null>(null);
  const [page, setPage] = useState(1);
  const [hasMoreCreations, setHasMoreCreations] = useState(true);

  const { getResonantFeed, likeCreation, shareCreation } = useMuseAPI();

  // Load feed data
  useEffect(() => {
    loadFeedData();
  }, [userId, filters, page]);

  // Load feed statistics
  useEffect(() => {
    loadFeedStats();
  }, [userId]);

  const loadFeedData = async () => {
    try {
      setLoading(true);
      const response = await getResonantFeed(userId, {
        ...filters,
        page,
        limit: 10,
        searchQuery: searchQuery.trim() || undefined
      });

      if (page === 1) {
        setCreations(response.creations);
      } else {
        setCreations(prev => [...prev, ...response.creations]);
      }

      setHasMoreCreations(response.hasMore);
      setError(null);
    } catch (err) {
      setError('Failed to load resonant feed');
      console.error('Feed loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadFeedStats = async () => {
    try {
      // Mock stats for now - would come from API
      setFeedStats({
        totalCreations: 1247,
        averageResonance: 0.78,
        topMuses: [
          { muse: 'ERATO', count: 342 },
          { muse: 'CALLIOPE', count: 289 },
          { muse: 'URANIA', count: 234 }
        ],
        trendingThemes: [
          { theme: 'nature', growth: 23 },
          { theme: 'cosmos', growth: 18 },
          { theme: 'mathematics', growth: 15 }
        ]
      });
    } catch (err) {
      console.error('Failed to load feed stats:', err);
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setPage(1);
    setCreations([]);
  };

  const handleFilterChange = (newFilters: Partial<FilterOptions>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setPage(1);
    setCreations([]);
  };

  const handleLike = async (creationId: string) => {
    try {
      await likeCreation(creationId, userId);
      setCreations(prev => 
        prev.map(creation => 
          creation.id === creationId 
            ? { ...creation, likes: creation.likes + 1, isLiked: true }
            : creation
        )
      );
    } catch (err) {
      console.error('Failed to like creation:', err);
    }
  };

  const handleShare = async (creationId: string, platform: string) => {
    try {
      await shareCreation(creationId, platform);
      setCreations(prev => 
        prev.map(creation => 
          creation.id === creationId 
            ? { ...creation, shares: creation.shares + 1 }
            : creation
        )
      );
    } catch (err) {
      console.error('Failed to share creation:', err);
    }
  };

  const loadMoreCreations = () => {
    if (!loading && hasMoreCreations) {
      setPage(prev => prev + 1);
    }
  };

  const filteredCreations = useMemo(() => {
    return creations.filter(creation => {
      if (searchQuery && !creation.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !creation.contentPreview.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [creations, searchQuery]);

  const sortOptions = [
    { value: 'resonance', label: 'Highest Resonance' },
    { value: 'recent', label: 'Most Recent' },
    { value: 'fitness', label: 'Mathematical Fitness' },
    { value: 'popular', label: 'Most Popular' }
  ];

  const formTypeOptions = [
    { value: 'all', label: 'All Forms' },
    { value: 'sonnet', label: 'Sonnets' },
    { value: 'haiku', label: 'Haiku' },
    { value: 'villanelle', label: 'Villanelles' },
    { value: 'free_verse', label: 'Free Verse' }
  ];

  const themeOptions = [
    { value: 'all', label: 'All Themes' },
    { value: 'nature', label: 'Nature' },
    { value: 'love', label: 'Love' },
    { value: 'cosmos', label: 'Cosmos' },
    { value: 'time', label: 'Time' },
    { value: 'mathematics', label: 'Mathematics' }
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Feed Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Sparkles className="h-6 w-6 text-purple-500" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Resonant Feed
            </h2>
          </div>
          {feedStats && (
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {feedStats.totalCreations.toLocaleString()} discoveries
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <TrendingUp className="h-5 w-5 text-green-500" />
          <span className="text-sm font-medium text-green-600 dark:text-green-400">
            {feedStats?.averageResonance ? `${(feedStats.averageResonance * 100).toFixed(1)}% avg resonance` : 'Loading...'}
          </span>
        </div>
      </div>

      {/* Search and Filters */}
      <Card className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm border-purple-200 dark:border-purple-800">
        <CardContent className="p-4">
          <div className="flex flex-col space-y-4">
            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search discoveries..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            {/* Filter Controls */}
            <div className="flex flex-wrap gap-4">
              <select
                value={filters.sortBy}
                onChange={(e) => handleFilterChange({ sortBy: e.target.value as any })}
                className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
              >
                {sortOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <select
                value={filters.formType}
                onChange={(e) => handleFilterChange({ formType: e.target.value })}
                className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
              >
                {formTypeOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <select
                value={filters.theme}
                onChange={(e) => handleFilterChange({ theme: e.target.value })}
                className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500"
              >
                {themeOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-600 dark:text-gray-400">
                  Min Fitness:
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={filters.minFitness}
                  onChange={(e) => handleFilterChange({ minFitness: parseFloat(e.target.value) })}
                  className="w-20"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400 min-w-[3rem]">
                  {(filters.minFitness * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feed Stats */}
      {feedStats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 border-purple-300 dark:border-purple-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Top Muses</p>
                  <div className="space-y-1">
                    {feedStats.topMuses.slice(0, 3).map((muse, index) => (
                      <div key={muse.muse} className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${
                          index === 0 ? 'bg-gold' : index === 1 ? 'bg-silver' : 'bg-bronze'
                        }`} />
                        <span className="text-sm font-medium">{muse.muse}</span>
                        <span className="text-xs text-gray-500">({muse.count})</span>
                      </div>
                    ))}
                  </div>
                </div>
                <Sparkles className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500/20 to-blue-500/20 border-green-300 dark:border-green-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Trending Themes</p>
                  <div className="space-y-1">
                    {feedStats.trendingThemes.slice(0, 3).map((theme) => (
                      <div key={theme.theme} className="flex items-center space-x-2">
                        <TrendingUp className="h-3 w-3 text-green-500" />
                        <span className="text-sm font-medium capitalize">{theme.theme}</span>
                        <span className="text-xs text-green-600">+{theme.growth}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <TrendingUp className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 border-blue-300 dark:border-blue-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Average Resonance</p>
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {(feedStats.averageResonance * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500">
                    Across {feedStats.totalCreations.toLocaleString()} discoveries
                  </p>
                </div>
                <Heart className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Feed Content */}
      <div className="space-y-4">
        {error && (
          <Card className="bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800">
            <CardContent className="p-4">
              <p className="text-red-700 dark:text-red-400">{error}</p>
            </CardContent>
          </Card>
        )}

        <AnimatePresence>
          {filteredCreations.map((creation, index) => (
            <motion.div
              key={creation.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <CreationCard
                creation={creation}
                currentUserId={userId}
                onLike={() => handleLike(creation.id)}
                onShare={(platform) => handleShare(creation.id, platform)}
                className="hover:shadow-lg transition-all duration-300"
              />
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <div className="flex justify-center py-8">
            <div className="flex items-center space-x-3">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full"
              />
              <span className="text-gray-600 dark:text-gray-400">
                Discovering resonant creations...
              </span>
            </div>
          </div>
        )}

        {!loading && filteredCreations.length === 0 && (
          <Card className="bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700">
            <CardContent className="p-8 text-center">
              <Sparkles className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No resonant discoveries found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Try adjusting your filters or exploring different themes to find mathematical poetry that resonates with your frequency signature.
              </p>
              <button
                onClick={() => {
                  setFilters({
                    formType: 'all',
                    minFitness: 0,
                    theme: 'all',
                    sortBy: 'resonance',
                    timeRange: 'week'
                  });
                  setSearchQuery('');
                }}
                className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
              >
                Reset Filters
              </button>
            </CardContent>
          </Card>
        )}

        {!loading && hasMoreCreations && filteredCreations.length > 0 && (
          <div className="flex justify-center py-4">
            <button
              onClick={loadMoreCreations}
              className="px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors flex items-center space-x-2"
            >
              <span>Load More Discoveries</span>
              <motion.div
                animate={{ y: [0, -2, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <Sparkles className="h-4 w-4" />
              </motion.div>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};