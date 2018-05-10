//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __Value2SignalArray_LAYERDATA_GPU_H__
#define __Value2SignalArray_LAYERDATA_GPU_H__

#include"Value2SignalArray_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Value2SignalArray_LayerData_GPU : public Value2SignalArray_LayerData_Base
	{
		friend class Value2SignalArray_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Value2SignalArray_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Value2SignalArray_LayerData_GPU();


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