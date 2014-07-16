__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 
__kernel void paracinza(
        __read_only image2d_t imgEnt,
        __write_only image2d_t imgSaida,
	const float th
    ) {
 
    const int2 pos = {get_global_id(0), get_global_id(1)};
    float4 pixel = read_imagef(imgEnt,sampler,pos);

    float media = (pixel.x+pixel.y+pixel.z)/3;

    if( media < th){
	write_imagef(imgSaida,pos,(float4)(255.f,0.f,0.f,0.f));
    }else{
	write_imagef(imgSaida,pos,pixel);
    }
    
 
}


__kernel void sub(__read_only image2d_t imgEnt1,
        __read_only image2d_t imgEnt2,
        __write_only image2d_t imgDiff,
	__global float* sum){

	int x = get_global_id(0);
	int y = get_global_id(1);

	const int2 pos = {x, y};
	float4 pixel1 = read_imagef(imgEnt1,sampler,pos);
	float4 pixel2 = read_imagef(imgEnt2,sampler,pos);

	float4 diff = sqrt((pixel1-pixel2)*(pixel1-pixel2));

	write_imagef(imgDiff,pos,diff);
	
	
	sum[y] += (diff.x+diff.y+diff.z); 
	
}

__kernel void copiar(
        __read_only image2d_t imgEnt,
        __write_only image2d_t imgSaida
    ) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
    uint4 pixel = read_imageui(imgEnt,sampler,pos);

    write_imageui(imgSaida,pos,pixel);
}
