//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MergeInput_LAYERDATA_GPU_H__
#define __MergeInput_LAYERDATA_GPU_H__

#include"MergeInput_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeInput_LayerData_GPU : public MergeInput_LayerData_Base
	{
		friend class MergeInput_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MergeInput_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MergeInput_LayerData_GPU();


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