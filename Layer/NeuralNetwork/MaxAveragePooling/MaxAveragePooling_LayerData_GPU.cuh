//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MaxAveragePooling_LAYERDATA_GPU_H__
#define __MaxAveragePooling_LAYERDATA_GPU_H__

#include"MaxAveragePooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MaxAveragePooling_LayerData_GPU : public MaxAveragePooling_LayerData_Base
	{
		friend class MaxAveragePooling_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MaxAveragePooling_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MaxAveragePooling_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif