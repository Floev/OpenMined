using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor
    {
	    private void Sort(SortOrder order = SortOrder.Ascending)
		{
			switch (order)
			{
				case SortOrder.Descending:
					base.Sort(data, SortIntDeascendingHelper.Get);
					break;
				case SortOrder.Ascending:
				default:
					base.Sort(data, SortIntAscendingHelper.Get);
					break;
			}
		}

		protected class SortIntAscendingHelper : IComparer<int>
		{
			static public SortIntAscendingHelper Get
			{
				get { 
					if (_instance == null) 
					{ 
						_instance = new SortIntAscendingHelper();
					}
					return _instance;
				}
			}
			static private SortIntAscendingHelper _instance = null;

			public int Compare(int x, int y)
			{
				if (x > y)
				{
					return 1;
				}

				if (y > x)
				{
					return -1;
				}

				return 0;
			}
		}

		protected class SortIntDeascendingHelper : IComparer<int>
		{
			static public SortIntDeascendingHelper Get
			{
				get
				{
					if (_instance == null)
					{
						_instance = new SortIntDeascendingHelper();
					}
					return _instance;
				}
			}
			static private SortIntDeascendingHelper _instance = null;

			public int Compare(int x, int y)
			{
				if (x > y)
				{
					return -1;
				}

				if (y > x)
				{
					return 1;
				}

				return 0;
			}
		}
    }
}

