#!/usr/bin/env python3

import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)

async def test_manager():
    try:
        from fintelligence_ai.knowledge.ingestion import KnowledgeBaseManager
        
        print("Creating manager...")
        manager = KnowledgeBaseManager()
        
        print("Initializing manager...")
        await manager.initialize()
        print("✅ Manager initialized successfully")
        
        print("Getting stats...")
        stats = await manager.get_knowledge_base_stats()
        print("✅ Stats:", stats)
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_manager()) 