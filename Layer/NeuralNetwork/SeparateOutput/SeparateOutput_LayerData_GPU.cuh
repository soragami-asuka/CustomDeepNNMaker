//======================================
// �o�͐M���������C���[�̃f�[�^
//======================================
#ifndef __SEPARATEOUTPUT_LAYERDATA_GPU_H__
#define __SEPARATEOUTPUT_LAYERDATA_GPU_H__

#include"SeparateOutput_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class SeparateOutput_LayerData_GPU : public SeparateOutput_LayerData_Base
	{
		friend class SeparateOutput_GPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		SeparateOutput_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~SeparateOutput_LayerData_GPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif