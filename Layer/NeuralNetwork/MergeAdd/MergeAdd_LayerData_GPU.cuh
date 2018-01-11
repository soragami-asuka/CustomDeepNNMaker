//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MergeAdd_LAYERDATA_GPU_H__
#define __MergeAdd_LAYERDATA_GPU_H__

#include"MergeAdd_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeAdd_LayerData_GPU : public MergeAdd_LayerData_Base
	{
		friend class MergeAdd_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MergeAdd_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MergeAdd_LayerData_GPU();


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