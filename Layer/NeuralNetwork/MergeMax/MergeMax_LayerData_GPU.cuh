//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MergeMax_LAYERDATA_GPU_H__
#define __MergeMax_LAYERDATA_GPU_H__

#include"MergeMax_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeMax_LayerData_GPU : public MergeMax_LayerData_Base
	{
		friend class MergeMax_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MergeMax_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MergeMax_LayerData_GPU();


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