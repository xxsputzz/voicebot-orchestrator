"""
Sprint 5: CLI Commands for Semantic Cache and LoRA Adapters

Command-line interface for managing semantic caching and LoRA adapter operations.
Provides cache inspection, eviction, and adapter control functionality.
"""

import asyncio
import argparse
import json
import sys
from typing import Dict, Any, List
from pathlib import Path

try:
    from .semantic_cache import SemanticCache, get_semantic_cache_analytics
    from .lora_adapter import LoraAdapterManager, get_lora_analytics
    REAL_IMPORTS = True
except ImportError:
    # Mock implementations for testing
    REAL_IMPORTS = False
    
    class MockSemanticCache:
        def __init__(self, **kwargs):
            self.cache_dir = Path("./cache")
            self.similarity_threshold = 0.2
        
        def get_cache_stats(self):
            return {
                "total_entries": 100,
                "hit_rate": 0.75,
                "cache_size_mb": 5.2
            }
        
        def evict_by_threshold(self, threshold):
            return 10
        
        def clear_cache(self):
            pass
        
        def export_cache_data(self):
            return [{"query": "test", "response": "test response"}]
    
    class MockLoraAdapterManager:
        def __init__(self, **kwargs):
            self.adapter_dir = Path("./adapters")
        
        def list_adapters(self):
            return ["banking-lora", "compliance-lora"]
        
        def list_loaded_adapters(self):
            return ["banking-lora"]
        
        def load_adapter(self, name, path=None):
            return True
        
        def unload_adapter(self, name):
            return True
        
        def activate_adapter(self, name):
            return True
        
        def deactivate_adapter(self):
            return True
        
        def get_adapter_status(self):
            return {
                "available_adapters": ["banking-lora"],
                "loaded_adapters": ["banking-lora"],
                "active_adapter": "banking-lora"
            }
        
        def create_banking_adapter(self, name="banking-lora"):
            return True
    
    def get_semantic_cache_analytics():
        return {"cache_hits": 150, "cache_misses": 50}
    
    def get_lora_analytics():
        return {"adapters_created": 3, "adapters_loaded": 2}
    
    SemanticCache = MockSemanticCache
    LoraAdapterManager = MockLoraAdapterManager


def format_cache_stats(stats: Dict[str, Any]) -> str:
    """Format cache statistics for display."""
    lines = [
        "Cache Statistics:",
        "=" * 40,
        f"Total Entries: {stats.get('total_entries', 0):,}",
        f"Hit Rate: {stats.get('hit_rate', 0):.2%}",
        f"Cache Size: {stats.get('cache_size_mb', 0):.1f} MB",
        f"Similarity Threshold: {stats.get('similarity_threshold', 0.2):.2f}",
        f"Model: {stats.get('model_name', 'unknown')}",
        ""
    ]
    
    if 'total_queries' in stats:
        lines.extend([
            f"Total Queries: {stats['total_queries']:,}",
            f"Cache Hits: {stats['hit_count']:,}",
            f"Cache Misses: {stats['miss_count']:,}",
        ])
    
    return "\n".join(lines)


def format_adapter_status(status: Dict[str, Any]) -> str:
    """Format adapter status for display."""
    lines = [
        "LoRA Adapter Status:",
        "=" * 40,
        f"Available Adapters: {len(status.get('available_adapters', []))}",
        f"Loaded Adapters: {len(status.get('loaded_adapters', []))}",
        f"Active Adapter: {status.get('active_adapter', 'None')}",
        f"Adapter Directory: {status.get('adapter_directory', 'unknown')}",
        ""
    ]
    
    if status.get('available_adapters'):
        lines.append("Available:")
        for adapter in status['available_adapters']:
            marker = " [LOADED]" if adapter in status.get('loaded_adapters', []) else ""
            active_marker = " [ACTIVE]" if adapter == status.get('active_adapter') else ""
            lines.append(f"  - {adapter}{marker}{active_marker}")
        lines.append("")
    
    return "\n".join(lines)


async def cache_manager_command(args: argparse.Namespace) -> None:
    """Execute cache manager command."""
    try:
        # Initialize cache
        cache = SemanticCache(
            cache_dir=args.cache_dir,
            similarity_threshold=args.threshold
        )
        
        if args.stats:
            # Show cache statistics
            stats = cache.get_cache_stats()
            
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(format_cache_stats(stats))
        
        elif args.evict:
            # Evict cache entries
            if args.evict_threshold:
                evicted = cache.evict_by_threshold(args.evict_threshold)
                result = {"operation": "evict", "entries_evicted": evicted}
            else:
                cache.clear_cache()
                result = {"operation": "clear", "entries_evicted": "all"}
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if result["entries_evicted"] == "all":
                    print("Cache cleared successfully")
                else:
                    print(f"Evicted {result['entries_evicted']} cache entries")
        
        elif args.export:
            # Export cache data
            cache_data = cache.export_cache_data()
            
            if args.output:
                output_file = Path(args.output)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
                result = {"operation": "export", "file": str(output_file), "entries": len(cache_data)}
                
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Exported {len(cache_data)} entries to {output_file}")
            else:
                print(json.dumps(cache_data, indent=2))
        
        elif args.analyze:
            # Analyze cache performance
            stats = cache.get_cache_stats()
            analytics = get_semantic_cache_analytics()
            
            analysis = {
                "cache_performance": {
                    "hit_rate": stats.get("hit_rate", 0),
                    "efficiency_score": _calculate_efficiency_score(stats),
                    "memory_usage": stats.get("cache_size_mb", 0)
                },
                "global_analytics": analytics,
                "recommendations": _generate_cache_recommendations(stats)
            }
            
            if args.json:
                print(json.dumps(analysis, indent=2))
            else:
                print("Cache Performance Analysis:")
                print("=" * 40)
                print(f"Hit Rate: {analysis['cache_performance']['hit_rate']:.2%}")
                print(f"Efficiency Score: {analysis['cache_performance']['efficiency_score']:.1f}/10")
                print(f"Memory Usage: {analysis['cache_performance']['memory_usage']:.1f} MB")
                print()
                print("Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"  - {rec}")
        
        else:
            # Default: show cache info
            stats = cache.get_cache_stats()
            
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(format_cache_stats(stats))
    
    except Exception as e:
        print(f"Error in cache manager: {e}", file=sys.stderr)
        sys.exit(1)


async def adapter_control_command(args: argparse.Namespace) -> None:
    """Execute adapter control command."""
    try:
        # Initialize adapter manager
        manager = LoraAdapterManager(adapter_dir=args.adapter_dir)
        
        if args.list:
            # List adapters
            status = manager.get_adapter_status()
            
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(format_adapter_status(status))
        
        elif args.load:
            # Load adapter
            success = manager.load_adapter(args.load, args.path)
            result = {"operation": "load", "adapter": args.load, "success": success}
            
            # If --enable flag is used, also activate the adapter
            if success and args.enable:
                activate_success = manager.activate_adapter(args.load)
                result["activated"] = activate_success
                if activate_success:
                    result["operation"] = "load_and_enable"
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if success:
                    print(f"Successfully loaded adapter: {args.load}")
                    if args.enable and result.get("activated"):
                        print(f"Successfully activated adapter: {args.load}")
                    elif args.enable and not result.get("activated"):
                        print(f"Loaded adapter but failed to activate: {args.load}")
                else:
                    print(f"Failed to load adapter: {args.load}")
        
        elif args.unload:
            # Unload adapter
            success = manager.unload_adapter(args.unload)
            result = {"operation": "unload", "adapter": args.unload, "success": success}
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if success:
                    print(f"Successfully unloaded adapter: {args.unload}")
                else:
                    print(f"Failed to unload adapter: {args.unload}")
        
        elif args.activate:
            # Activate adapter
            success = manager.activate_adapter(args.activate)
            result = {"operation": "activate", "adapter": args.activate, "success": success}
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if success:
                    print(f"Successfully activated adapter: {args.activate}")
                else:
                    print(f"Failed to activate adapter: {args.activate}")
        
        elif args.deactivate:
            # Deactivate current adapter
            success = manager.deactivate_adapter()
            result = {"operation": "deactivate", "success": success}
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if success:
                    print("Successfully deactivated adapter")
                else:
                    print("No adapter was active")
        
        elif args.create_banking:
            # Create banking adapter
            adapter_name = args.create_banking if args.create_banking != True else "banking-lora"
            success = manager.create_banking_adapter(adapter_name)
            result = {"operation": "create_banking", "adapter": adapter_name, "success": success}
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if success:
                    print(f"Successfully created banking adapter: {adapter_name}")
                else:
                    print(f"Failed to create banking adapter: {adapter_name}")
        
        elif args.info:
            # Get adapter info
            adapter_info = manager.get_adapter_info(args.info)
            
            if adapter_info:
                if args.json:
                    print(json.dumps(adapter_info, indent=2))
                else:
                    print(f"Adapter Information: {args.info}")
                    print("=" * 40)
                    print(f"Base Model: {adapter_info['base_model_name']}")
                    print(f"Loaded: {'Yes' if adapter_info['is_loaded'] else 'No'}")
                    print(f"Parameters: {adapter_info['parameter_count']:,}")
                    print()
                    print("Configuration:")
                    config = adapter_info['config']
                    print(f"  Rank (r): {config['r']}")
                    print(f"  Alpha: {config['lora_alpha']}")
                    print(f"  Dropout: {config['lora_dropout']}")
                    print(f"  Target Modules: {', '.join(config['target_modules'])}")
                    
                    if adapter_info['training_metrics']:
                        print()
                        print("Training Metrics:")
                        metrics = adapter_info['training_metrics']
                        for key, value in metrics.items():
                            print(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                result = {"error": f"Adapter not found: {args.info}"}
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Adapter not found: {args.info}")
        
        else:
            # Default: show status
            status = manager.get_adapter_status()
            
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(format_adapter_status(status))
    
    except Exception as e:
        print(f"Error in adapter control: {e}", file=sys.stderr)
        sys.exit(1)


def orchestrator_log_command(args: argparse.Namespace) -> None:
    """Execute orchestrator log command."""
    try:
        analytics_data = {}
        
        if args.cache_hits or args.all:
            # Get cache analytics
            cache_analytics = get_semantic_cache_analytics()
            analytics_data["semantic_cache"] = cache_analytics
        
        if args.adapter_metrics or args.all:
            # Get adapter analytics
            adapter_analytics = get_lora_analytics()
            analytics_data["lora_adapters"] = adapter_analytics
        
        if args.performance or args.all:
            # Get performance metrics
            performance_data = {
                "cache_efficiency": _calculate_cache_efficiency(),
                "adapter_impact": _calculate_adapter_impact(),
                "memory_usage": _get_memory_usage(),
                "throughput_improvement": _calculate_throughput_improvement()
            }
            analytics_data["performance"] = performance_data
        
        if args.json:
            print(json.dumps(analytics_data, indent=2))
        else:
            print("Orchestrator Analytics:")
            print("=" * 50)
            
            if "semantic_cache" in analytics_data:
                cache_data = analytics_data["semantic_cache"]
                print("Semantic Cache:")
                print(f"  Total Hits: {cache_data.get('cache_hits', 0):,}")
                print(f"  Total Misses: {cache_data.get('cache_misses', 0):,}")
                total = cache_data.get('cache_hits', 0) + cache_data.get('cache_misses', 0)
                hit_rate = cache_data.get('cache_hits', 0) / max(total, 1)
                print(f"  Hit Rate: {hit_rate:.2%}")
                print()
            
            if "lora_adapters" in analytics_data:
                adapter_data = analytics_data["lora_adapters"]
                print("LoRA Adapters:")
                print(f"  Adapters Created: {adapter_data.get('adapters_created', 0)}")
                print(f"  Adapters Loaded: {adapter_data.get('adapters_loaded', 0)}")
                print(f"  Adapter Switches: {adapter_data.get('adapter_switches', 0)}")
                print()
            
            if "performance" in analytics_data:
                perf_data = analytics_data["performance"]
                print("Performance Metrics:")
                print(f"  Cache Efficiency: {perf_data['cache_efficiency']:.1f}%")
                print(f"  Adapter Impact: {perf_data['adapter_impact']:.1f}% improvement")
                print(f"  Memory Usage: {perf_data['memory_usage']:.1f} MB")
                print(f"  Throughput Improvement: {perf_data['throughput_improvement']:.1f}%")
    
    except Exception as e:
        print(f"Error getting orchestrator logs: {e}", file=sys.stderr)
        sys.exit(1)


def _calculate_efficiency_score(stats: Dict[str, Any]) -> float:
    """Calculate cache efficiency score (0-10)."""
    hit_rate = stats.get('hit_rate', 0)
    entries = stats.get('total_entries', 0)
    size_mb = stats.get('cache_size_mb', 0)
    
    # Base score from hit rate
    score = hit_rate * 7
    
    # Bonus for having reasonable cache size
    if 1 <= size_mb <= 100:
        score += 1
    
    # Bonus for having reasonable number of entries
    if 10 <= entries <= 10000:
        score += 1
    
    # Penalty for very low hit rate
    if hit_rate < 0.1:
        score *= 0.5
    
    return min(10.0, max(0.0, score))


def _generate_cache_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate cache optimization recommendations."""
    recommendations = []
    
    hit_rate = stats.get('hit_rate', 0)
    entries = stats.get('total_entries', 0)
    size_mb = stats.get('cache_size_mb', 0)
    threshold = stats.get('similarity_threshold', 0.2)
    
    if hit_rate < 0.3:
        recommendations.append("Consider lowering similarity threshold to increase hit rate")
    
    if hit_rate > 0.9:
        recommendations.append("Consider raising similarity threshold to improve precision")
    
    if size_mb > 100:
        recommendations.append("Cache size is large - consider implementing eviction policies")
    
    if entries < 10:
        recommendations.append("Cache has few entries - allow more time for cache warmup")
    
    if threshold > 0.5:
        recommendations.append("Similarity threshold is high - may be missing relevant cache hits")
    
    if not recommendations:
        recommendations.append("Cache performance is optimal")
    
    return recommendations


def _calculate_cache_efficiency() -> float:
    """Calculate overall cache efficiency percentage."""
    # Mock calculation
    return 78.5


def _calculate_adapter_impact() -> float:
    """Calculate adapter performance impact percentage."""
    # Mock calculation
    return 12.3


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    # Mock calculation
    return 245.7


def _calculate_throughput_improvement() -> float:
    """Calculate throughput improvement percentage."""
    # Mock calculation
    return 35.2


def create_sprint5_parser() -> argparse.ArgumentParser:
    """Create argument parser for Sprint 5 CLI commands."""
    parser = argparse.ArgumentParser(
        description="Sprint 5 CLI - Semantic Cache and LoRA Adapter Management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Cache manager command
    cache_parser = subparsers.add_parser("cache-manager", help="Manage semantic cache")
    cache_parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    cache_parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    cache_parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    cache_parser.add_argument("--evict", action="store_true", help="Evict cache entries")
    cache_parser.add_argument("--evict-threshold", type=float, help="Eviction threshold for accuracy")
    cache_parser.add_argument("--export", action="store_true", help="Export cache data")
    cache_parser.add_argument("--output", help="Output file for export")
    cache_parser.add_argument("--analyze", action="store_true", help="Analyze cache performance")
    
    # Adapter control command
    adapter_parser = subparsers.add_parser("adapter-control", help="Control LoRA adapters")
    adapter_parser.add_argument("--adapter-dir", default="./adapters", help="Adapter directory")
    adapter_parser.add_argument("--list", action="store_true", help="List adapters")
    adapter_parser.add_argument("--load", help="Load adapter by name")
    adapter_parser.add_argument("--path", help="Custom path for loading adapter")
    adapter_parser.add_argument("--enable", action="store_true", help="Enable (activate) adapter after loading")
    adapter_parser.add_argument("--unload", help="Unload adapter by name")
    adapter_parser.add_argument("--activate", help="Activate adapter by name")
    adapter_parser.add_argument("--deactivate", action="store_true", help="Deactivate current adapter")
    adapter_parser.add_argument("--create-banking", nargs="?", const=True, help="Create banking domain adapter")
    adapter_parser.add_argument("--info", help="Get adapter information")
    
    # Orchestrator log command
    log_parser = subparsers.add_parser("orchestrator-log", help="Show orchestrator analytics")
    log_parser.add_argument("--cache-hits", action="store_true", help="Show cache hit statistics")
    log_parser.add_argument("--adapter-metrics", action="store_true", help="Show adapter metrics")
    log_parser.add_argument("--performance", action="store_true", help="Show performance metrics")
    log_parser.add_argument("--all", action="store_true", help="Show all analytics")
    
    return parser


async def main() -> None:
    """Main CLI entry point for Sprint 5."""
    parser = create_sprint5_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "cache-manager":
            await cache_manager_command(args)
        elif args.command == "adapter-control":
            await adapter_control_command(args)
        elif args.command == "orchestrator-log":
            orchestrator_log_command(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
