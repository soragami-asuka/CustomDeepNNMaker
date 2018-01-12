//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MergeAverage_LAYERDATA_CPU_H__
#define __MergeAverage_LAYERDATA_CPU_H__

#include"MergeAverage_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeAverage_LayerData_CPU : public MergeAverage_LayerData_Base
	{
		friend class MergeAverage_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MergeAverage_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MergeAverage_LayerData_CPU();


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