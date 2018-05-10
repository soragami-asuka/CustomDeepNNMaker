//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __ProbabilityArray2Value_LAYERDATA_GPU_H__
#define __ProbabilityArray2Value_LAYERDATA_GPU_H__

#include"ProbabilityArray2Value_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ProbabilityArray2Value_LayerData_GPU : public ProbabilityArray2Value_LayerData_Base
	{
		friend class ProbabilityArray2Value_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		ProbabilityArray2Value_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~ProbabilityArray2Value_LayerData_GPU();


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