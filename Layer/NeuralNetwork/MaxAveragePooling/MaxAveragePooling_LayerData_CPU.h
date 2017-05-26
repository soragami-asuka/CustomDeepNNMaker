//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MaxAveragePooling_LAYERDATA_CPU_H__
#define __MaxAveragePooling_LAYERDATA_CPU_H__

#include"MaxAveragePooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MaxAveragePooling_LayerData_CPU : public MaxAveragePooling_LayerData_Base
	{
		friend class MaxAveragePooling_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MaxAveragePooling_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MaxAveragePooling_LayerData_CPU();


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