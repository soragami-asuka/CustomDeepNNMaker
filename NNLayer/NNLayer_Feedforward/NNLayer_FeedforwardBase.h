//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include<NNLayerInterface/INNLayer.h>

#include<vector>

#include"NNLayer_Feedforward_DATA.hpp"

namespace Gravisbell {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class NNLayer_FeedforwardBase : public Gravisbell::NeuralNetwork::INNLayer
	{
	protected:
		GUID guid;

		ILayerConfig* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
		ILayerConfig* pLearnData;		/**< �w�K�ݒ���`�����R���t�B�O�N���X */

		NNLayer_Feedforward::LayerStructure layerStructure;	/**< ���C���[�\�� */
		NNLayer_Feedforward::LearnDataStructure learnData;	/**< �w�K�ݒ� */

		std::vector<IOutputLayer*> lppInputFromLayer;		/**< ���͌����C���[�̃��X�g */
		std::vector<IInputLayer*>  lppOutputToLayer;	/**< �o�͐惌�C���[�̃��X�g */

		unsigned int batchSize;	/**< �o�b�`�T�C�Y */

	public:
		/** �R���X�g���N�^ */
		NNLayer_FeedforwardBase(GUID guid);

		/** �f�X�g���N�^ */
		virtual ~NNLayer_FeedforwardBase();

		//===========================
		// ���C���[����
		//===========================
	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		unsigned int GetLayerKindBase()const;

		/** ���C���[�ŗL��GUID���擾���� */
		ErrorCode GetGUID(GUID& o_guid)const;

		/** ���C���[�̎�ގ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		ErrorCode GetLayerCode(GUID& o_layerCode)const;

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		unsigned int GetBatchSize()const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const ILayerConfig& config);
		/** ���C���[�̐ݒ�����擾���� */
		const ILayerConfig* GetLayerConfig()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		unsigned int GetUseBufferByteCount()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual const IODataStruct GetInputDataStruct()const = 0;

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const;


		//===========================
		// �ŗL�֐�
		//===========================
	public:
		/** �j���[���������擾���� */
		unsigned int GetNeuronCount()const;
	};

}
}