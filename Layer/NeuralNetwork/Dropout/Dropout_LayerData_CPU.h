//======================================
// 活性化関数のレイヤーデータ
//======================================
#ifndef __Dropout_LAYERDATA_CPU_H__
#define __Dropout_LAYERDATA_CPU_H__

#include"Dropout_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Dropout_LayerData_CPU : public Dropout_LayerData_Base
	{
		friend class Dropout_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Dropout_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Dropout_LayerData_CPU();


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