//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __GlobalAveragePooling_LAYERDATA_GPU_H__
#define __GlobalAveragePooling_LAYERDATA_GPU_H__

#include"GlobalAveragePooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class GlobalAveragePooling_LayerData_GPU : public GlobalAveragePooling_LayerData_Base
	{
		friend class GlobalAveragePooling_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		GlobalAveragePooling_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~GlobalAveragePooling_LayerData_GPU();


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