//======================================
// 畳み込みニューラルネットワークのレイヤーデータ
//======================================
#ifndef __UpSampling_LAYERDATA_GPU_H__
#define __UpSampling_LAYERDATA_GPU_H__


#include"UpSampling_LayerData_Base.h"

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class UpSampling_LayerData_GPU : public UpSampling_LayerData_Base
	{
		friend class UpSampling_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		UpSampling_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~UpSampling_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif