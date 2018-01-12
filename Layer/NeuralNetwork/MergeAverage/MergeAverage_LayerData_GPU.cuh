//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MergeAverage_LAYERDATA_GPU_H__
#define __MergeAverage_LAYERDATA_GPU_H__

#include"MergeAverage_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeAverage_LayerData_GPU : public MergeAverage_LayerData_Base
	{
		friend class MergeAverage_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MergeAverage_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MergeAverage_LayerData_GPU();


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