//======================================
// �o�b�`���K���̃��C���[�f�[�^
//======================================
#ifndef __ExponentialNormalization_DATA_BASE_H__
#define __ExponentialNormalization_DATA_BASE_H__

#include<Layer/ILayerData.h>
#include<Layer/NeuralNetwork/IOptimizer.h>

#include<vector>

#include"ExponentialNormalization_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class ExponentialNormalization_LayerData_Base : public ILayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[�f�[�^���ʗp��GUID */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		ExponentialNormalization::LayerStructure layerStructure;	/**< ���C���[�\�� */

		U64 learnTime;	/**< �w�K�� */

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		ExponentialNormalization_LayerData_Base(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		virtual ~ExponentialNormalization_LayerData_Base();


		//===========================
		// ���ʏ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U64 GetUseBufferByteCount()const;


	public:
		//===========================
		// ���C���[�\��
		//===========================
		/** ���̓f�[�^�\�����g�p�\���m�F����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
		bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** �o�̓f�[�^�\�����擾����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
		IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** �����o�͂��\�����m�F���� */
		bool CheckCanHaveMultOutputLayer(void);

		

		//===========================
		// �ŗL�֐�
		//===========================
	public:

		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================		
	public:
		/** �I�v�e�B�}�C�U�[��ύX���� */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif