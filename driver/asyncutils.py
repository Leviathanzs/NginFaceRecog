import asyncio

async def GatherAsync(future_to_exec):
        return await asyncio.gather(*[task() for task in future_to_exec])