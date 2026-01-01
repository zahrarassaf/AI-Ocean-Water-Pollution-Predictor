"""
Parallel processing utilities
"""

import concurrent.futures
from functools import partial
from typing import Callable, List, Any, Optional, Dict
import multiprocessing as mp
import numpy as np
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import logging

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Utility for parallel processing"""
    
    def __init__(self, 
                 n_workers: Optional[int] = None,
                 backend: str = 'multiprocessing',
                 use_dask: bool = False):
        """
        Initialize parallel processor
        
        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes
        backend : str
            Backend to use: 'multiprocessing', 'threading', 'dask'
        use_dask : bool
            Whether to use Dask distributed
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.backend = backend
        self.use_dask = use_dask
        self.client = None
        
        if use_dask and backend == 'dask':
            self._init_dask_cluster()
    
    def _init_dask_cluster(self):
        """Initialize Dask cluster"""
        try:
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=1,
                memory_limit='4GB',
                processes=True,
                silence_logs=logging.ERROR
            )
            self.client = Client(cluster)
            logger.info(f"Dask cluster initialized with {self.n_workers} workers")
        except Exception as e:
            logger.warning(f"Could not initialize Dask cluster: {e}")
            self.use_dask = False
    
    def parallel_map(self,
                     func: Callable,
                     data: List[Any],
                     chunksize: int = 1,
                     desc: str = "Processing") -> List[Any]:
        """
        Parallel map function
        
        Parameters
        ----------
        func : callable
            Function to apply
        data : list
            Input data
        chunksize : int
            Chunk size for processing
        desc : str
            Description for logging
            
        Returns
        -------
        list
            Processed results
        """
        logger.info(f"{desc} with {self.n_workers} workers")
        
        if self.use_dask and self.backend == 'dask':
            # Use Dask for parallel processing
            lazy_results = []
            for item in data:
                lazy_result = dask.delayed(func)(item)
                lazy_results.append(lazy_result)
            
            results = dask.compute(*lazy_results, scheduler='processes')
            return list(results)
        
        else:
            # Use concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_workers
            ) as executor:
                results = list(executor.map(func, data, chunksize=chunksize))
            
            return results
    
    def parallel_apply(self,
                       dataset: np.ndarray,
                       func: Callable,
                       axis: int = 0,
                       **kwargs) -> np.ndarray:
        """
        Apply function in parallel along array axis
        
        Parameters
        ----------
        dataset : np.ndarray
            Input array
        func : callable
            Function to apply
        axis : int
            Axis to apply function along
            
        Returns
        -------
        np.ndarray
            Processed array
        """
        if axis >= dataset.ndim:
            raise ValueError(f"Axis {axis} out of bounds for {dataset.ndim}D array")
        
        # Split array along axis
        slices = np.array_split(dataset, self.n_workers, axis=axis)
        
        # Create partial function with kwargs
        func_partial = partial(func, **kwargs)
        
        # Process slices in parallel
        results = self.parallel_map(func_partial, slices)
        
        # Concatenate results
        return np.concatenate(results, axis=axis)
    
    def batch_process(self,
                      data: List[Any],
                      batch_size: int,
                      func: Callable,
                      **kwargs) -> List[Any]:
        """
        Process data in batches
        
        Parameters
        ----------
        data : list
            Input data
        batch_size : int
            Batch size
        func : callable
            Processing function
        
        Returns
        -------
        list
            Processed results
        """
        n_batches = (len(data) + batch_size - 1) // batch_size
        results = []
        
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(data))
            batch = data[batch_start:batch_end]
            
            logger.info(f"Processing batch {i+1}/{n_batches} "
                       f"({len(batch)} items)")
            
            batch_results = self.parallel_map(func, batch, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def close(self):
        """Clean up resources"""
        if self.client:
            self.client.close()
            logger.info("Dask cluster closed")

def parallelize_large_array(operation: str,
                           array: np.ndarray,
                           func: Callable,
                           chunk_size: Optional[tuple] = None,
                           overlap: int = 0,
                           **kwargs) -> np.ndarray:
    """
    Parallelize operations on large arrays
    
    Parameters
    ----------
    operation : str
        Operation type: 'apply', 'reduce', 'transform'
    array : np.ndarray
        Input array
    func : callable
        Function to apply
    chunk_size : tuple, optional
        Chunk size for processing
    overlap : int
        Overlap between chunks for seamless processing
    **kwargs
        Additional arguments for func
        
    Returns
    -------
    np.ndarray
        Processed array
    """
    if chunk_size is None:
        # Default chunk size based on array shape
        chunk_size = tuple(max(1, s // 4) for s in array.shape)
    
    # Convert to Dask array
    dask_array = da.from_array(array, chunks=chunk_size)
    
    if operation == 'apply':
        # Apply function to each chunk
        result = dask_array.map_blocks(func, **kwargs)
    
    elif operation == 'reduce':
        # Reduce operation
        result = dask_array.reduction(
            func,
            aggregate=func,
            **kwargs
        )
    
    elif operation == 'transform':
        # Transform with overlap for seamless processing
        result = dask_array.map_overlap(
            func,
            depth=overlap,
            boundary='reflect',
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Compute result
    return result.compute()

class ProgressTracker:
    """Track progress of parallel operations"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.lock = mp.Lock()
    
    def update(self, n: int = 1):
        """Update progress"""
        with self.lock:
            self.completed_tasks += n
            progress = (self.completed_tasks / self.total_tasks) * 100
            logger.info(f"Progress: {progress:.1f}% "
                       f"({self.completed_tasks}/{self.total_tasks})")
    
    def get_progress(self) -> float:
        """Get current progress"""
        return (self.completed_tasks / self.total_tasks) * 100
