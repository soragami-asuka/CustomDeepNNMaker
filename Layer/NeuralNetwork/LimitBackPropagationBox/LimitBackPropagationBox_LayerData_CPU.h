//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __LimitBackPropagationBox_LAYERDATA_CPU_H__
#define __LimitBackPropagationBox_LAYERDATA_CPU_H__

#include"LimitBackPropagationBox_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class LimitBackPropagationBox_LayerData_CPU : public LimitBackPropagationBox_LayerData_Base
	{
		friend class LimitBackPropagationBox_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		LimitBackPropagationBox_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~LimitBackPropagationBox_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif