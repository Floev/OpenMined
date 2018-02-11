using System;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
	public partial class BaseTensor<T>
	{
		public enum SortOrder { Ascending, Descending };

		///<summary> 
		/// If the partition size is fewer than 16 elements, it uses an insertion sort algorithm.
		/// If the number of partitions exceeds 2 * LogN, where N is the range of the input array, it uses a Heapsort algorithm.
		/// Otherwise, it uses a Quicksort algorithm.
		///</summary>
		// https://msdn.microsoft.com/en-us/library/6tf1f0bc(v=vs.110).aspx
		// This implementation performs an unstable sort; 
		// that is, if two elements are equal, their order might not be preserved.In contrast, a stable sort preserves the order of elements that are equal.
		protected void Sort(T[] elements, IComparer<T> comparer)
		{
			Array.Sort(elements, comparer);
		}
	}
}