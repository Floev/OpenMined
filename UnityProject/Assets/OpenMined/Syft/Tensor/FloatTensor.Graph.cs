using System;
using System.Collections.Generic;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private List<int> creators;
        private string creation_op;
        private List<int> children_indices; // children -> counts
	    private List<int> children_counts; // children -> counts
	    private int sibling;

        public void InitAutograd()
        {
//			if(!autograd) {
            autograd = true;
	        InitGraph();
//			}
        }

	    public void InitGraph()
	    {
		    creators = new List<int>();
		    children_indices = new List<int>();
		    children_counts = new List<int>();
	    }

	    public void ResetAutogradCounts()
	    {
		    for (int i = 0; i < children_counts.Count; i++)
		    {
			    children_counts[i] = 0;
		    }		    
	    }

        // hook autograd many tensor parents
        public void HookAutograd(ref FloatTensor result, string creation_op, params FloatTensor[] args)
        {
            if (autograd)
            {
                result.InitAutograd();
                result.creation_op = creation_op;
//                result.creators.Add(this);
//                children.Add(result.Id, 0);
                foreach (FloatTensor x in args)
                {
 //                   result.creators.Add(x);
                    //can't pass params by ref, need to do outside
                    //x.children.Add(result.Id, 0);
                };
            }
        }

        // hook autograd one parents - one scalar
        public FloatTensor HookAutograd(ref FloatTensor result, float x, string creation_op, bool inline)
        {
	        if (inline)
		        return this;
	        
	        bool autograd_pre_initialized = false;
            
	        if (result == null)
	        {
		        bool child_pre_initialized = false;
		        int child_index = 0;
                //remove outermost if clause?
		        if (children_indices.Count > 0)
		        {
			        for (int i = 0; i < children_indices.Count; i++)
			        {
				        FloatTensor temp = factory.Get(children_indices[i]);
				        
				        if (temp.creation_op == creation_op)
				        {
					        if (temp.creators.Count > 1)
					        {
						        FloatTensor temp2 = factory.Get(temp.creators[1]);
						        if (temp2.data[0] == x)
						        {
							        //if (temp2.autograd == temp.autograd)
							        //{
								        child_pre_initialized = true;
								        child_index = children_indices[i];
							        //}
						        }
					        }
				        }
			        }
		        }
		        
		        if (child_pre_initialized)
		        {
			        autograd_pre_initialized = true;
			        result = factory.Get(child_index);
			        result.Zero_();
			        //Debug.Log("Graph:93:Fetching Tensor:" + result.id + " with creation_op:" + result.creation_op + " called under creation op:" + creation_op);
		        }
		        else
		        {
			        result = factory.Create(_shape: this.shape,
				        _data: data,
				        _dataBuffer: dataBuffer,
				        _shapeBuffer: shapeBuffer,
				        _shader: shader,
				        _copyData: true,
				        _dataOnGpu: dataOnGpu,
				        _autograd: autograd,
				        _keepgrads: keepgrads,
				        _creation_op: creation_op);			        
			        //Debug.Log("Graph:109:Creating Tensor:" + result.id + " with creation_op:" + result.creation_op);
		        }
	        }

	        if (autograd_pre_initialized)
	        {
		        ResetAutogradCounts();
		        result.ResetAutogradCounts();
	        }
	        else
	        {
		        FloatTensor new_child = factory.Create(
			        _shape: new int[] {1},
			        _data: new float[] {x},
			        _dataBuffer: dataBuffer,
			        _shapeBuffer: shapeBuffer,
			        _shader: shader,
			        _copyData: true,
			        _dataOnGpu: dataOnGpu,
			        _autograd: autograd,
			        _keepgrads: keepgrads,
			        _creation_op: creation_op);
                		        
		        result.InitGraph();
		        result.creators.Add(id);
		        result.creators.Add(new_child.id);
		        result.creation_op = creation_op;
		        
		        children_indices.Add(result.Id);
		        children_counts.Add(0);
	        }

	        return result;
        }

		// hook autograd two parents
		public FloatTensor HookAutograd(ref FloatTensor result, ref FloatTensor x, string creation_op, bool inline=false, int[] resultShape= null)
		{

			if (inline)
				return this;
		
			// checks to see if the input has been seen previously. If so, then it assumes
			// that we should just use the previous computation graph instead of initializing
			// a new result. The assumption here is that if the same tensors are used to perform
			// the same operation, then they should output to the same memory instead of allocating
			// new memory.
			bool autograd_pre_initialized = false;

			if (result == null)
            {
				bool child_pre_initialized = false;
				int child_index = 0;
                //delete the outermost if clause?
				if (children_indices.Count > 0)
				{
					// iterate through children
					for (int i = 0; i < children_indices.Count; i++)
					{
						FloatTensor temp = factory.Get(children_indices[i]);
						
						// if a child was created using the same op as the one currently being called
						// and the child was also created using the same tensor as x
						// then it's exactly the same operation and we can re-use variables.
						if (temp.creation_op == creation_op && temp.creators.Contains(x.id))
						{
							child_pre_initialized = true;
							child_index = children_indices[i];
						}
					}			        
				}
				
				if (child_pre_initialized)
				{
					//Debug.Log("Id:" + this.id + " Children:" + this.children_indices.Count);
					autograd_pre_initialized = true;
					result = factory.Get(child_index);
					result.Zero_();
					//Debug.Log("Graph:148:Fetching Tensor:" + result.id + " with creation_op:" + result.creation_op + " called under creation op:" + creation_op);
				}
				else
				{
					if (resultShape != null)
					{
						// initializes an empty tensor with new shape
						result = factory.Create(
							_shape: resultShape,
							_dataOnGpu: dataOnGpu,
							_autograd: x.autograd && autograd,
							_keepgrads: keepgrads,
							_creation_op: creation_op);
						//Debug.Log("Graph:187:Creating Tensor:" + result.id + " with creation_op:" + result.creation_op);
					}
					else
					{
						// initializes an empty tensor with identical shape
						result = factory.Create(
							_shape: shape,
							_data: data,
							_dataBuffer: dataBuffer,
							_shapeBuffer: shapeBuffer,
							_shader: shader,
							_copyData: true,
							_dataOnGpu: dataOnGpu,
							_autograd: x.autograd && autograd, // if either tensor doesn't have gradients
							_keepgrads: keepgrads,			   // neither does the result. This might not end up being
							_creation_op: creation_op);        // a good decision in the long run. We'll see.
						//Debug.Log("Graph:202:Creating Tensor:" + result.id + " with creation_op:" + result.creation_op);
					}
					
					// this is sortof a backup check. In theory, the result tensor should have been 
					// initialized correctly.
					if (dataOnGpu)
						result.Gpu(shader);
				}
			}

			if (autograd_pre_initialized)
			{
				ResetAutogradCounts();
				result.ResetAutogradCounts();
				x.ResetAutogradCounts();
			}
			else
			{
				result.InitGraph();
				result.creators.Add(id);
				result.creators.Add(x.id);
				result.creation_op = creation_op;

				children_indices.Add(result.Id);
				children_counts.Add(0);

				x.children_indices.Add(result.Id);
				x.children_counts.Add(0);

				sibling = x.id;
			}

            return result;
		}

		// hook autograd single parent
		public FloatTensor HookAutograd(ref FloatTensor result, string creation_op, bool inline=false, int[] resultShape = null)
        {
			if (inline)
				return this;
			
			bool autograd_pre_initialized = false;
			//Debug.Log("Id:" + this.id + " Children:" + this.children.Count);
			if (result == null)
			{				
				bool child_pre_initialized = false;
				int child_index = 0;
//Can we delete this outermost if clause?
				if (this.children_indices.Count > 0)
				{
					for (int i = 0; i < this.children_indices.Count; i++)
					{
						if (factory.Get(children_indices[i]).creation_op == creation_op)
						{
							child_pre_initialized = true;
							child_index = children_indices[i];
						}
					}
				}
		        
				if (child_pre_initialized)
				{
					autograd_pre_initialized = true;
					result = factory.Get(child_index);
					result.Zero_();
					//Debug.Log("Graph:237:Fetching Tensor:" + result.id + " with creation_op:" + result.creation_op + " called under creation op:" + creation_op);
				}
				else
				{
					if (resultShape != null)
					{
						result = factory.Create(
							_shape: resultShape,
							_dataOnGpu: dataOnGpu,
							_autograd: autograd,
							_keepgrads: keepgrads,
							_creation_op: creation_op);
						//Debug.Log("Graph:187:Creating Tensor:" + result.id + " with creation_op:" + result.creation_op);
					}
					else
					{
						result = factory.Create(
							_shape: this.shape,
							_data: data,
							_dataBuffer: dataBuffer,
							_shapeBuffer: shapeBuffer,
							_shader: shader,
							_copyData: true,
							_dataOnGpu: dataOnGpu,
							_autograd: autograd,
							_keepgrads: keepgrads,
							_creation_op: creation_op);

						//Debug.Log("Graph:254:Creating Tensor:" + result.id + " with creation_op:" + result.creation_op);
					}
				}
			}
			
			if (autograd_pre_initialized)
			{
				this.ResetAutogradCounts();
				result.ResetAutogradCounts();
			}
			else 
			{				
				result.InitGraph ();
				result.creators.Add (this.id);
				result.creation_op = creation_op;

				children_indices.Add(result.Id);
				children_counts.Add(0);
			}

			return result;
		}
    }
}