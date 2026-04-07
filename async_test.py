import asyncio
import time

async def brew_tea(name, brew_time):
    print(f"Started brewing {name}...")
    # While this tea brews, the event loop goes to start the next tea
    await asyncio.sleep(brew_time) 
    print(f"Finished brewing {name}!")

async def main():
    start_time = time.time()
    
    # Run all three coroutines concurrently
    await asyncio.gather(
        brew_tea("Green Tea", 2),
        brew_tea("Black Tea", 3),
        brew_tea("Chamomile", 1)
    )
    
    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

asyncio.run(main())